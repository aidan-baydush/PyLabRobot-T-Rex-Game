# T-Rex Game for OT-2
import os
import numpy as np
import pandas as pd
import random
import math
from typing import Iterator, Tuple, List, AsyncIterator
import asyncio
import sys

from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.backend import LiquidHandlerBackend
from pylabrobot.liquid_handling.backends import LiquidHandlerChatterboxBackend
from pylabrobot.resources import (
    Plate, 
    corning_96_wellplate_360ul_flat, 
    OTDeck, 
    Deck, 
    opentrons_96_tiprack_300ul,
    opentrons_96_tiprack_1000ul,
    axygen_1_reservoir_90ml,
    agilent_1_reservoir_290ml,
    set_tip_tracking, 
    set_volume_tracking, 
    set_cross_contamination_tracking, 
    TipRack
)
from pylabrobot.visualizer.visualizer import Visualizer
import pylabrobot.resources.functional as F


# ============ DISPLAY FUNCTIONS (from counting notebook) ============
def get_digit_wells(digit_value, col_start):
    """
    Get wells for a digit spanning 3 columns (col_start to col_start+2)
    Rows A-E form the 7-segment display
    """
    col_left = col_start
    col_mid = col_start + 1
    col_right = col_start + 2
    
    segments = {
        0: [f'A{col_left}', f'A{col_mid}', f'A{col_right}',  # top
            f'B{col_left}', f'C{col_left}', f'D{col_left}',  # left side
            f'B{col_right}', f'C{col_right}', f'D{col_right}',  # right side
            f'E{col_left}', f'E{col_mid}', f'E{col_right}'],  # bottom
        
        1: [f'A{col_right}', f'B{col_right}', f'C{col_right}', 
            f'D{col_right}', f'E{col_right}'],  # right side only
        
        2: [f'A{col_left}', f'A{col_mid}', f'A{col_right}',  # top
            f'B{col_right}', f'C{col_right}',  # top-right
            f'C{col_mid}',  # middle
            f'D{col_left}', f'E{col_left}',  # bottom-left
            f'E{col_left}', f'E{col_mid}', f'E{col_right}'],  # bottom
        
        3: [f'A{col_left}', f'A{col_mid}', f'A{col_right}',  # top
            f'B{col_right}', f'C{col_right}', f'D{col_right}', f'E{col_right}',  # right side
            f'C{col_mid}',  # middle
            f'E{col_left}', f'E{col_mid}'],  # bottom
        
        4: [f'A{col_left}', f'B{col_left}', f'C{col_left}',  # top-left
            f'C{col_mid}',  # middle
            f'A{col_right}', f'B{col_right}', f'C{col_right}', f'D{col_right}', f'E{col_right}'],  # right side
        
        5: [f'A{col_left}', f'A{col_mid}', f'A{col_right}',  # top
            f'B{col_left}', f'C{col_left}',  # top-left
            f'C{col_mid}',  # middle
            f'D{col_right}', f'E{col_right}',  # bottom-right
            f'E{col_left}', f'E{col_mid}'],  # bottom
        
        6: [f'A{col_left}', f'A{col_mid}', f'A{col_right}',  # top
            f'B{col_left}', f'C{col_left}', f'D{col_left}', f'E{col_left}',  # left side
            f'C{col_mid}',  # middle
            f'D{col_right}', f'E{col_right}',  # bottom-right
            f'E{col_mid}'],  # bottom
        
        7: [f'A{col_left}', f'A{col_mid}', f'A{col_right}',  # top
            f'B{col_right}', f'C{col_right}', f'D{col_right}', f'E{col_right}'],  # right side
        
        8: [f'A{col_left}', f'A{col_mid}', f'A{col_right}',  # top
            f'B{col_left}', f'B{col_right}',  # row B sides
            f'C{col_left}', f'C{col_mid}', f'C{col_right}',  # row C (middle segment)
            f'D{col_left}', f'D{col_right}',  # row D sides
            f'E{col_left}', f'E{col_mid}', f'E{col_right}'],  # bottom
        
        9: [f'A{col_left}', f'A{col_mid}', f'A{col_right}',  # top
            f'B{col_left}', f'B{col_right}',  # row B both sides
            f'C{col_left}', f'C{col_mid}', f'C{col_right}',  # middle
            f'D{col_right}', f'E{col_right}'],  # right side bottom
    }
    
    return segments[digit_value]


# Create tip position tracking
tip_positions = []
for row in 'ABCDEFGH':
    for col in range(1, 13):
        tip_positions.append(f'{row}{col}')

current_tip_index = 0
current_rack_index = 0


async def get_next_tip_spot(lh):
    """Get next available tip from tip racks"""
    global current_tip_index, current_rack_index
    
    tip_racks = [lh.deck.get_resource(f'tip_rack_{i}') for i in range(9)]
    
    if current_rack_index >= len(tip_racks):
        raise RuntimeError("Out of tips!")
    
    current_rack = tip_racks[current_rack_index]
    tip_position = tip_positions[current_tip_index]
    tip_spot = current_rack[tip_position]
    
    current_tip_index += 1
    if current_tip_index >= 96:
        current_tip_index = 0
        current_rack_index += 1
    
    return tip_spot


async def display_number(lh, number: int):
    """Display a number on the 7-segment display using wells"""
    if number < 0 or number > 9999:
        raise ValueError("Number must be between 0 and 9999")
    
    game_plate = lh.deck.get_resource("game")
    res = lh.deck.get_resource("res")
    
    # Display positions: ones, tens, hundreds, thousands (right to left)
    digits_str = str(number).zfill(4)[::-1]
    digit_positions = [10, 7, 4, 1]
    
    # Track current display state
    current_display = [set() for _ in range(4)]
    if hasattr(lh, '_current_display'):
        current_display = lh._current_display
    
    for idx, (digit_char, col_start) in enumerate(zip(digits_str, digit_positions)):
        # Skip leading zeros
        if number < 10 ** idx:
            continue
            
        digit = int(digit_char)
        new_wells = set(get_digit_wells(digit, col_start))
        old_wells = current_display[idx]
        
        wells_to_remove = old_wells - new_wells
        wells_to_add = new_wells - old_wells
        
        # Remove old segments
        for well_id in wells_to_remove:
            tip_spot = await get_next_tip_spot(lh)
            await lh.pick_up_tips(tip_spots=tip_spot, use_channels=[0])
            await lh.aspirate(resources=[game_plate.get_well(well_id)], vols=[300])
            await lh.dispense(resources=[res.get_well("A1")], vols=[300])
            await lh.discard_tips()
        
        # Add new segments
        for well_id in wells_to_add:
            tip_spot = await get_next_tip_spot(lh)
            await lh.pick_up_tips(tip_spots=tip_spot, use_channels=[0])
            await lh.aspirate(resources=[res.get_well("A1")], vols=[300])
            await lh.dispense(resources=[game_plate.get_well(well_id)], vols=[300])
            await lh.discard_tips()
        
        current_display[idx] = new_wells
    
    lh._current_display = current_display


# ============ GAME FUNCTIONS ============
def set_deck() -> Deck:
    """Set up the OT-2 deck with game plate, tip racks, and reservoir"""
    deck = OTDeck(name='deck')
    plate = corning_96_wellplate_360ul_flat(name='game')
    deck.assign_child_at_slot(plate, 1)
    
    # Add 9 tip racks
    tip_racks = [opentrons_96_tiprack_300ul(f'tip_rack_{i}') for i in range(9)]
    for i in range(9):
        deck.assign_child_at_slot(tip_racks[i], i+3)
    
    # Add reservoir
    res = axygen_1_reservoir_90ml('res')
    deck.assign_child_at_slot(res, 2)

    return deck


class Dino:
    def __init__(self, deck: Deck, lh: LiquidHandler, plate: Plate, res):
        self.board: List[str] = []            # contains occupied wells
        self.deck = deck
        self.lh = lh
        self.plate = plate
        self.res = res
        self.dino_pos: List[str] = []
        self.obstacle_vol = 300
        self.dino_vol = 300
        self.score = 0

    async def setup(self):
        """Initialize the game by placing the dino on the plate"""
        await self._pipette_res_to_well(dest='G2', vols=self.dino_vol)
        await self._pipette_res_to_well(dest='H2', vols=self.dino_vol)
        self.dino_pos.append('G2')
        self.dino_pos.append('H2')
        
        # Initialize display to 0
        await display_number(self.lh, 0)

    async def _pipette_plate_to_plate(self, dest: str, source: str, vols: int):
        """Method to pipette from the plate to the plate"""
        tip_spot = await get_next_tip_spot(self.lh)
        await self.lh.pick_up_tips(tip_spots=tip_spot, use_channels=[0])
        await self.lh.aspirate(resources=[self.plate.get_well(source)], vols=[vols], use_channels=[0])
        await self.lh.dispense(resources=[self.plate.get_well(dest)], vols=[vols], use_channels=[0])
        await self.lh.return_tips()
        self.board.append(dest)
        self.board.remove(source)

    async def _pipette_res_to_well(self, dest: str, vols: int):
        """Method to pipette from the reservoir to the well"""
        tip_spot = await get_next_tip_spot(self.lh)
        await self.lh.pick_up_tips(tip_spots=tip_spot, use_channels=[0])
        await self.lh.aspirate(resources=[self.res.get_well('A1')], vols=[vols], use_channels=[0])
        await self.lh.dispense(resources=[self.plate.get_well(dest)], vols=[vols], use_channels=[0])
        await self.lh.return_tips()
        self.board.append(dest)

    async def _pipette_well_to_res(self, source: str, vols: int):
        """Method to pipette from plate to the reservoir"""
        tip_spot = await get_next_tip_spot(self.lh)
        await self.lh.pick_up_tips(tip_spots=tip_spot, use_channels=[0])
        await self.lh.aspirate(resources=[self.plate.get_well(source)], vols=[vols], use_channels=[0])
        await self.lh.dispense(resources=[self.res.get_well('A1')], vols=[vols], use_channels=[0])
        await self.lh.return_tips()
        self.board.remove(source)

    async def push_board(self):
        """Move all obstacles one column to the left"""
        no_dino = self.board.copy()
        for pos in self.dino_pos:
            if pos in no_dino:
                no_dino.remove(pos)
        
        # Sort by column number (left to right) to process leftmost obstacles first
        no_dino.sort(key=lambda x: int(x[1:]))
        
        for pos in no_dino:
            current_col = int(pos[1:])
            
            if current_col == 1:  # Reached leftmost column
                await self._pipette_well_to_res(source=pos, vols=self.obstacle_vol)
                self.score += 1
            else:
                # Move one column to the left
                next_col = current_col - 1
                next_pos = f'{pos[0]}{next_col}'
                
                # Check collision with dino
                if next_pos in self.dino_pos:
                    await self.game_over()
                
                # Move obstacle left by one column
                await self._pipette_plate_to_plate(source=pos, dest=next_pos, vols=self.obstacle_vol)

    async def dino_jump(self):
        """Make the dino jump (move from H2 to F2)"""
        if not self.dino_pos == ['H2', 'G2']:
            await self.dino_return()
        if 'H2' in self.dino_pos:
            await self._pipette_plate_to_plate(dest='F2', source='H2', vols=self.dino_vol)
            self.dino_pos.remove('H2')
            self.dino_pos.append('F2')

    async def dino_squat(self):
        """Make the dino squat (remove G2)"""
        if not self.dino_pos == ['H2', 'G2']:
            await self.dino_return()
        if 'G2' in self.dino_pos:
            await self._pipette_well_to_res(source='G2', vols=self.dino_vol)
            self.dino_pos.remove('G2')

    async def dino_return(self):
        """Return dino to normal position"""
        if 'F2' in self.dino_pos and 'H2' not in self.dino_pos:
            # Return from jump
            await self._pipette_plate_to_plate(dest='H2', source='F2', vols=self.dino_vol)
            self.dino_pos.remove('F2')
            self.dino_pos.append('H2')
        
        if 'G2' not in self.dino_pos and 'H2' in self.dino_pos:
            # Return from squat
            await self._pipette_res_to_well(dest='G2', vols=self.dino_vol)
            self.dino_pos.append('G2')

    async def new_obstacle(self):
        """Spawn a new random obstacle at column 12"""
        obstacle_type = int(random.random() * 3)
        loc = {0: 'H12', 1: 'G12', 2: 'F12'}
        await self._pipette_res_to_well(dest=loc[obstacle_type], vols=self.obstacle_vol)

    async def game_over(self):
        """End the game and display final score"""
        print('\n' + '='*50)
        print('GAME OVER!')
        print(f'Final Score: {self.score}')
        print('='*50)
        sys.exit(0)


async def run_game(dino_obj, turns):
    """Main game loop"""
    await dino_obj.setup()
    print("\n" + "="*50)
    print("T-REX GAME - OT-2 Edition")
    print("="*50)
    print("Controls:")
    print("  w - Jump")
    print("  s - Squat")
    print("  [space] - Return to normal")
    print("="*50 + "\n")
    
    for i in range(turns):
        print(f"\nRound {i+1} | Score: {dino_obj.score}")
        
        # Get user input
        user_input = input('Your move (w/s/space): ')
        
        if user_input == 's':
            await dino_obj.dino_squat()
        elif user_input == 'w':
            await dino_obj.dino_jump()
        elif user_input == ' ':
            await dino_obj.dino_return()
        else:
            print("Invalid input! Use w, s, or space")
        
        # Push board and update display
        await dino_obj.push_board()
        
        # Add new obstacle every 3 rounds
        if i % 3 == 0:
            await dino_obj.new_obstacle()
        
        # Update score display
        await display_number(dino_obj.lh, dino_obj.score)
    
    print("\nGame completed!")
    sys.exit(0)


async def main():
    """Main entry point"""
    # Enable tracking
    set_tip_tracking(enabled=True)
    set_volume_tracking(enabled=True)
    set_cross_contamination_tracking(enabled=True)
    
    # Set up deck and liquid handler
    deck = set_deck()
    lh = LiquidHandler(backend=LiquidHandlerChatterboxBackend(), deck=deck)
    await lh.setup()
    
    # Set up visualizer
    vis = Visualizer(resource=lh)
    await vis.setup()
    
    # Initialize reservoir with liquid
    res = lh.deck.get_resource("res")
    res.get_item('A1').tracker.set_liquids([(None, 90000)])  # 90mL reservoir capacity
    
    # Create game object
    dino = Dino(
        deck=deck,
        lh=lh,
        plate=deck.get_resource('game'),
        res=res
    )
    
    print("\nVisualizer is running!")
    print("Open your browser to: http://127.0.0.1:1338")
    print("Watch the game unfold on the virtual deck!\n")
    
    # Run game for 100 turns
    N = 100
    await run_game(dino, N)


if __name__ == '__main__':
    asyncio.run(main())
