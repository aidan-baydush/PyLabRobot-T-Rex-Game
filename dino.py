# @aidan-baydush [Aidan Baydush] @ ... input name bowen
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
from pylabrobot.resources import Plate, corning_96_wellplate_360ul_flat, OTDeck, Deck, opentrons_96_tiprack_300ul, axygen_1_reservoir_90ml, set_tip_tracking, set_volume_tracking, set_cross_contamination_tracking, TipRack
from pylabrobot.visualizer.visualizer import Visualizer
import pylabrobot.resources.functional as F


# some function headers
def set_deck() -> Deck:
    deck = OTDeck(name='deck')
    plate = corning_96_wellplate_360ul_flat(name='game')
    deck.assign_child_at_slot(plate, 1)
    tip_racks = [opentrons_96_tiprack_300ul(f'tip_rack_{i}') for i in range(9)]
    for i in range(9):
        # puts tip racks in 3-11
        deck.assign_child_at_slot(tip_racks[i], i+3)
    res = axygen_1_reservoir_90ml('res')
    deck.assign_child_at_slot(res, 2)

    return deck


def get_tip_generator(deck: Deck) -> AsyncIterator:
    # get all tip spots
    tip_racks = [deck.get_resource(f'tip_rack_{i}') for i in range(9)]

    tip_spots = F.get_all_tip_spots(tip_racks)

    # build a linear generator to get all new spots (from module 2)
    linear_generator = F.linear_tip_spot_generator(
        tip_spots=tip_spots,                      # the list of tip spots to use
        repeat=True,                              # repeat the tip spots if they run out
    )
    return linear_generator


class Dino:
    def __init__(self, tip_gen: AsyncIterator, deck: Deck, lh: LiquidHandler, plate: Plate, res):
        self.board: List[str] = []            # contains occupied wells
        self.tip_gen = tip_gen,
        self.deck = deck
        self.lh = lh
        self.plate = plate
        self.res = res
        self.dino_pos: List[str] = []
        self.obstacle_vol = 50
        self.dino_vol = 100
        self.score = 0

    async def setup(self):
        '''
            Adds dino to plate.
        '''
        await self._pipette_res_to_well(dest='G2', vols=self.dino_vol)
        await self._pipette_res_to_well(dest='H2', vols=self.dino_vol)
        self.dino_pos.append('G2')
        self.dino_pos.append('H2')

    async def _pipette_plate_to_plate(self, dest: str, source: str, vols: int):
        '''
            Method to pipette from the plate to the plate
        '''
        await self.lh.pick_up_tips(tip_spots=[await anext(self.tip_gen)], use_channels=[0])
        await self.lh.aspirate(resources=[self.plate.get_well(source)], vols=[vols], use_channels=[0])
        await self.lh.dispense(resources=[self.plate.get_well(dest)], vols=[vols], use_channels=[0])
        await self.lh.return_tips()
        self.board.append(dest)
        self.board.remove(source)

    async def _pipette_res_to_well(self, dest: str, vols: int):
        '''
            Method to pipette from the resivoir to the well. 
        '''
        await self.lh.pick_up_tips(tip_spots=[await anext(self.tip_gen)], use_channels=[0])
        await self.lh.aspirate(resources=[self.res.get_well('A1')], vols=[vols], use_channels=[0])
        await self.lh.dispense(resources=[self.plate.get_well(dest)], vols=[vols], use_channels=[0])
        await self.lh.return_tips()
        self.board.append(dest)

    async def _pipette_well_to_res(self, source: str, vols: int):
        '''
            Method to pipette from plate to the resivoir
        '''
        await self.lh.pick_up_tips(tip_spots=[await anext(self.tip_gen)], use_channels=[0])
        await self.lh.aspirate(resources=[self.plate.get_well(source)], vols=[vols], use_channels=[0])
        await self.lh.dispense(resources=[self.res.get_well('A1')], vols=[vols], use_channels=[0])
        await self.lh.return_tips()
        self.board.remove(source)

    async def push_board(self):
        no_dino = self.board.copy()
        for pos in self.dino_pos:                                           # dont move dino!
            no_dino.remove(pos)
        for pos in no_dino:
            if pos[-1] == '1':                                              # remove from board
                await self._pipette_well_to_res(source=pos, vols=self.obstacle_vol)
                self.score += 1
            else:
                # collision detection
                if f'{pos[0]}{int(pos[-1])-1}' in self.dino_pos:
                    self.game_over()                                        # end game
                # else, push all board obstacles over
                await self._pipette_plate_to_plate(source=pos, vols=self.obstacle_vol, dest=f'{pos[0]}{str(int(pos[-1])-1)}')

    async def dino_jump(self):
        await self._pipette_plate_to_plate(dest='F2', source='H2', vols=self.dino_vol)
        self.dino_pos.remove('H2')
        self.dino_pos.append('F2')

    async def dino_squat(self):
        await self._pipette_well_to_res(source='G2', vols=self.dino_vol)
        self.dino_pos.remove('G2')

    async def dino_return(self):
        if self.dino_pos == ['F2', 'G2']:
            await self._pipette_plate_to_plate(dest='H2', source='F2', vols=self.dino_vol)
        elif self.dino_pos == ['H2']:
            await self._pipette_res_to_well(dest='G2', vols=self.dino_vol)
        self.dino_pos = ['H2', 'G2']

    async def new_obstacle(self):
        obstacle_type = int(random.random() * 3)
        loc = {0: 'H12', 1: 'G12', 2: 'F12'}
        await self._pipette_res_to_well(dest=loc[obstacle_type], vols=self.obstacle_vol)

    def game_over(self):
        print('the game has ended. thank you for playing')
        print(f'score: {self.score}')
        sys.exit(0)


def run(dino_obj, turns):
    asyncio.run(dino_obj.setup())
    for i in range(turns):
        # user input
        input = input(
            'your move: s is for squat, w is for jump, and space is for staying.')
        match input:
            case 's':
                asyncio.run(dino_obj.dino_squat())
            case 'w':
                asyncio.run(dino_obj.dino_jump())
            case ' ':
                asyncio.run(dino_obj.dino_return())
        # now push board and every 3rd push add a new obstacle
        asyncio.run(dino_obj.push_board())
        if i % 3 == 0:
            asyncio.run(dino_obj.new_obstacle())
        # now change the score
        asyncio.run(display_number(lh, dino_obj.score))
    sys.exit()


if __name__ == '__main__':
    deck = set_deck()
    tip_gen = get_tip_generator(deck=deck)
    lh = LiquidHandler(backend=LiquidHandlerChatterboxBackend, deck=deck)
    dino = Dino(tip_gen=tip_gen, deck=deck, lh=lh,
                plate=deck.get_resource('game'), res=deck.get_resource('res'))
    N = 100
    run(dino, N)
