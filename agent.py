import os
import sys
import time
import traceback
import math
from collections.abc import Hashable

import chess
import chess.polyglot
import numpy as np
from numba import njit, uint64, int32, b1

# Evaluation Tables & Constants

MG_VALUE = {1: 82, 2: 337, 3: 365, 4: 477, 5: 1025, 6: 0}
EG_VALUE = {1: 94, 2: 281, 3: 297, 4: 512, 5: 936, 6: 0}

PHASE_INC = {1: 0, 2: 1, 3: 1, 4: 2, 5: 4, 6: 0}

MG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]

EG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]

MG_KNIGHT = [
   -167, -89, -34, -49,  61, -97, -15, -107,
    -73, -41,  72,  36,  23,  62,   7,  -17,
    -47,  60,  37,  65,  84, 129,  73,   44,
     -9,  17,  19,  53,  37,  69,  18,   22,
    -13,   4,  16,  13,  28,  19,  21,   -8,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
   -105, -21, -58, -33, -17, -28, -19,  -23,
]

EG_KNIGHT = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]

MG_BISHOP = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]

EG_BISHOP = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
]

MG_ROOK = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
]

EG_ROOK = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
]

MG_QUEEN = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]

EG_QUEEN = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]

MG_KING = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]

EG_KING = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

_MG_RAW = {1: MG_PAWN, 2: MG_KNIGHT, 3: MG_BISHOP, 4: MG_ROOK, 5: MG_QUEEN, 6: MG_KING}
_EG_RAW = {1: EG_PAWN, 2: EG_KNIGHT, 3: EG_BISHOP, 4: EG_ROOK, 5: EG_QUEEN, 6: EG_KING}

MG_TABLE: list[list[int]] = [[0] * 64]
EG_TABLE: list[list[int]] = [[0] * 64]
for _pt in range(1, 7):
    MG_TABLE.append([MG_VALUE[_pt] + v for v in _MG_RAW[_pt]])
    EG_TABLE.append([EG_VALUE[_pt] + v for v in _EG_RAW[_pt]])

# Numpy Arrays for Numba
mg_table_np = np.array(MG_TABLE, dtype=np.int32)
eg_table_np = np.array(EG_TABLE, dtype=np.int32)
phase_inc_np = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)

PASSED_MG = [0, 0, 2, 5, 12, 22, 40, 0]
PASSED_EG = [0, 10, 18, 34, 60, 100, 150, 0]
passed_mg_np = np.array(PASSED_MG, dtype=np.int32)
passed_eg_np = np.array(PASSED_EG, dtype=np.int32)

_KP_SCALE = [0, 0, 0, 1, 2, 3, 4, 0]
kp_scale_np = np.array(_KP_SCALE, dtype=np.int32)

KING_DIST: list[list[int]] = [
    [max(abs((a & 7) - (b & 7)), abs((a >> 3) - (b >> 3))) for b in range(64)]
    for a in range(64)
]
king_dist_np = np.array(KING_DIST, dtype=np.int32)

# Numba Bitboard Constants
BB_ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
NOT_FILE_A = np.uint64(0xFEFEFEFEFEFEFEFE)
NOT_FILE_H = np.uint64(0x7F7F7F7F7F7F7F7F)
OUTPOST_RANKS_W = np.uint64(0x0000FFFFFF000000)
OUTPOST_RANKS_B = np.uint64(0x000000FFFFFF0000)

# Search constants
INF = 1 << 30
MATE = 1000000
MATE_BOUND = MATE - 4096
MAX_PLY = 96

TT_BONUS = 1 << 22
CAPTURE_BONUS = 1 << 20
KILLER1_BONUS = (1 << 19) + 2
KILLER2_BONUS = (1 << 19) + 1
CHECK_BONUS = 1 << 18

SEE_VALUE = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000, None: 100}
EXACT, UPPER, LOWER = 0, 1, 2
TT_MAX_ENTRIES = 500000

START_DEPTH = 4
START_DEPTH_MIN_MS = 1500
UNDO_PENALTY = 14

WHITE = chess.WHITE
BLACK = chess.BLACK

class _Timeout(Exception):
    """Raised internally when the move deadline is reached."""

@njit(uint64(uint64, uint64, b1), cache=True)
def _passers_jit(own, enemy, white):
    if white:
        span = enemy >> np.uint64(8)
        span |= span >> np.uint64(8)
        span |= span >> np.uint64(16)
        span |= span >> np.uint64(32)
    else:
        span = enemy << np.uint64(8)
        span |= span << np.uint64(8)
        span |= span << np.uint64(16)
        span |= span << np.uint64(32)
    span |= ((span & NOT_FILE_H) << np.uint64(1)) | ((span & NOT_FILE_A) >> np.uint64(1))
    return own & ~span

@njit(int32(uint64), cache=True)
def popcount(n):
    count = 0
    while n:
        n &= n - np.uint64(1)
        count += 1
    return count

@njit(int32(
    uint64, uint64, uint64, uint64, uint64, uint64, 
    uint64, uint64, uint64, uint64, uint64, uint64, b1
), cache=True)
def _evaluate_jit(wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk, is_white_turn):
    mg = 0
    eg = 0
    phase = 0

    white_pieces = (wp, wn, wb, wr, wq, wk)
    black_pieces = (bp, bn, bb, br, bq, bk)

    wk_sq = -1
    bk_sq = -1

    for pt in range(6):
        w = white_pieces[pt]
        while w:
            lsb = w & -w
            sq = int(np.log2(float(lsb)))
            w ^= lsb
            idx = sq ^ 56
            mg += mg_table_np[pt + 1, idx]
            eg += eg_table_np[pt + 1, idx]
            phase += phase_inc_np[pt + 1]
            if pt == 5:
                wk_sq = sq

        b = black_pieces[pt]
        while b:
            lsb = b & -b
            sq = int(np.log2(float(lsb)))
            b ^= lsb
            mg -= mg_table_np[pt + 1, sq]
            eg -= eg_table_np[pt + 1, sq]
            phase += phase_inc_np[pt + 1]
            if pt == 5:
                bk_sq = sq

    w_pawn_attacks = ((wp & NOT_FILE_A) << np.uint64(7)) | ((wp & NOT_FILE_H) << np.uint64(9))
    b_pawn_attacks = ((bp & NOT_FILE_H) >> np.uint64(7)) | ((bp & NOT_FILE_A) >> np.uint64(9))

    # Passed Pawn Scaling (Halve bonus if unsupported/isolated)
    if wp or bp:
        w = _passers_jit(wp, bp, True)
        while w:
            lsb = w & -w
            sq = int(np.log2(float(lsb)))
            w ^= lsb
            r = sq >> 3
            
            is_supported = (lsb & w_pawn_attacks) or (wk_sq >= 0 and king_dist_np[wk_sq, sq] <= 2)
            mg_bonus = passed_mg_np[r]
            eg_bonus = passed_eg_np[r]
            if not is_supported:
                mg_bonus //= 2
                eg_bonus //= 2
                
            mg += mg_bonus
            eg += eg_bonus
            stop = sq + 8
            if stop < 64 and wk_sq >= 0 and bk_sq >= 0:
                eg += (2 * king_dist_np[bk_sq, stop] - king_dist_np[wk_sq, stop]) * kp_scale_np[r]

        b = _passers_jit(bp, wp, False)
        while b:
            lsb = b & -b
            sq = int(np.log2(float(lsb)))
            b ^= lsb
            r = 7 - (sq >> 3)
            
            is_supported = (lsb & b_pawn_attacks) or (bk_sq >= 0 and king_dist_np[bk_sq, sq] <= 2)
            mg_bonus = passed_mg_np[r]
            eg_bonus = passed_eg_np[r]
            if not is_supported:
                mg_bonus //= 2
                eg_bonus //= 2

            mg -= mg_bonus
            eg -= eg_bonus
            stop = sq - 8
            if stop >= 0 and wk_sq >= 0 and bk_sq >= 0:
                eg -= (2 * king_dist_np[wk_sq, stop] - king_dist_np[bk_sq, stop]) * kp_scale_np[r]

    b_front_span = bp >> np.uint64(8)
    b_front_span |= b_front_span >> np.uint64(8)
    b_front_span |= b_front_span >> np.uint64(16)
    b_front_span |= b_front_span >> np.uint64(32)
    b_attack_span = b_front_span | ((b_front_span & NOT_FILE_A) >> np.uint64(1)) | ((b_front_span & NOT_FILE_H) << np.uint64(1))
    
    w_front_span = wp << np.uint64(8)
    w_front_span |= w_front_span << np.uint64(8)
    w_front_span |= w_front_span << np.uint64(16)
    w_front_span |= w_front_span << np.uint64(32)
    w_attack_span = w_front_span | ((w_front_span & NOT_FILE_A) >> np.uint64(1)) | ((w_front_span & NOT_FILE_H) << np.uint64(1))

    w_outposts = w_pawn_attacks & (~b_attack_span) & OUTPOST_RANKS_W
    b_outposts = b_pawn_attacks & (~w_attack_span) & OUTPOST_RANKS_B

    w_ko_count = popcount(wn & w_outposts)
    if w_ko_count:
        bonus = w_ko_count * 30
        mg += bonus
        eg += bonus

    b_ko_count = popcount(bn & b_outposts)
    if b_ko_count:
        bonus = b_ko_count * 30
        mg -= bonus
        eg -= bonus

    if wk_sq >= 0:
        wk_file = wk_sq & 7
        enemy_pawn_storm = 0
        p = bp
        while p:
            lsb = p & -p
            p_sq = int(np.log2(float(lsb)))
            p_file = p_sq & 7
            p_rank = p_sq >> 3
            if abs(p_file - wk_file) <= 1 and p_rank <= 4:
                enemy_pawn_storm += (5 - p_rank) * 12
            p ^= lsb
        mg -= enemy_pawn_storm

    if bk_sq >= 0:
        bk_file = bk_sq & 7
        enemy_pawn_storm_w = 0
        p = wp
        while p:
            lsb = p & -p
            p_sq = int(np.log2(float(lsb)))
            p_file = p_sq & 7
            p_rank = p_sq >> 3
            if abs(p_file - bk_file) <= 1 and p_rank >= 3:
                enemy_pawn_storm_w += p_rank * 12
            p ^= lsb
        mg += enemy_pawn_storm_w

    u = wp | (wp << np.uint64(8))
    u |= u << np.uint64(16)
    u |= u << np.uint64(32)
    w_files = u | (u >> np.uint64(8))
    w_files |= w_files >> np.uint64(16)
    w_files |= w_files >> np.uint64(32)

    v = bp | (bp >> np.uint64(8))
    v |= v >> np.uint64(16)
    v |= v >> np.uint64(32)
    b_files = v | (v << np.uint64(8))
    b_files |= b_files << np.uint64(16)
    b_files |= b_files << np.uint64(32)

    if wp:
        d = popcount(wp & (u << np.uint64(8)))
        adj = ((w_files & NOT_FILE_H) << np.uint64(1)) | ((w_files & NOT_FILE_A) >> np.uint64(1))
        i = popcount(wp & ~adj)
        mg -= 8 * d + 12 * i
        eg -= 18 * d + 16 * i

    if bp:
        d = popcount(bp & (v >> np.uint64(8)))
        adj = ((b_files & NOT_FILE_H) << np.uint64(1)) | ((b_files & NOT_FILE_A) >> np.uint64(1))
        i = popcount(bp & ~adj)
        mg += 8 * d + 12 * i
        eg += 18 * d + 16 * i

    w_rooks_alone = wr & ~w_files
    if w_rooks_alone:
        semi = popcount(w_rooks_alone & b_files)
        opn = popcount(w_rooks_alone & ~b_files)
        mg += 26 * opn + 12 * semi
        eg += 14 * opn + 8 * semi

    b_rooks_alone = br & ~b_files
    if b_rooks_alone:
        semi = popcount(b_rooks_alone & w_files)
        opn = popcount(b_rooks_alone & ~w_files)
        mg -= 26 * opn + 12 * semi
        eg -= 14 * opn + 8 * semi

    if phase <= 8 and wk_sq >= 0 and bk_sq >= 0:
        w_center_dist = abs((wk_sq & 7) - 3.5) + abs((wk_sq >> 3) - 3.5)
        b_center_dist = abs((bk_sq & 7) - 3.5) + abs((bk_sq >> 3) - 3.5)
        eg += int((7.0 - w_center_dist) * 4)
        eg -= int((7.0 - b_center_dist) * 4)

    if phase <= 6 and (wp or bp):
        near = 0
        p = bp
        while p:
            lsb = p & -p
            sq = int(np.log2(float(lsb)))
            p ^= lsb
            near -= king_dist_np[wk_sq, sq]
        p = wp
        while p:
            lsb = p & -p
            sq = int(np.log2(float(lsb)))
            p ^= lsb
            near += king_dist_np[bk_sq, sq]
        eg += 3 * near

    if popcount(wb) >= 2:
        mg += 22
        eg += 40
    if popcount(bb) >= 2:
        mg -= 22
        eg -= 40

    if phase > 24:
        phase = 24
    score = (mg * phase + eg * (24 - phase)) // 24

    if is_white_turn:
        return score + 12
    return -score + 12


# --- Tactical Penalty Evaluation (Python Wrapper Additions) ---

PIN_PENALTY = {1: 4, 2: 18, 3: 18, 4: 28, 5: 45, 6: 0}
def _pin_score(board: chess.Board, color: chess.Color) -> int:
    penalty = 0
    pieces = board.knights | board.bishops | board.rooks | board.queens
    for sq in chess.SquareSet(pieces & board.occupied_co[color]):
        if board.is_pinned(color, sq):
            pt = board.piece_type_at(sq)
            if pt:
                penalty += PIN_PENALTY[pt]
    return penalty

KING_ZONE_WEIGHT = {2: 20, 3: 20, 4: 40, 5: 80}
def _king_zone_pressure(board: chess.Board, king_sq: int, king_color: chess.Color) -> int:
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]) | {king_sq}
    pressure = 0
    for sq in zone:
        for a_sq in board.attackers(not king_color, sq):
            pt = board.piece_type_at(a_sq)
            if pt:
                pressure += KING_ZONE_WEIGHT.get(pt, 0)
    return pressure

def evaluate(board: chess.Board) -> int:
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    
    score = _evaluate_jit(
        np.uint64(board.pawns & white), np.uint64(board.knights & white),
        np.uint64(board.bishops & white), np.uint64(board.rooks & white),
        np.uint64(board.queens & white), np.uint64(board.kings & white),
        np.uint64(board.pawns & black), np.uint64(board.knights & black),
        np.uint64(board.bishops & black), np.uint64(board.rooks & black),
        np.uint64(board.queens & black), np.uint64(board.kings & black),
        board.turn
    )

    w_pins = _pin_score(board, chess.WHITE)
    b_pins = _pin_score(board, chess.BLACK)
    
    w_king_pressure = 0
    b_king_pressure = 0
    if board.queens:
        w_king_sq = board.king(chess.WHITE)
        if w_king_sq is not None:
            w_king_pressure = _king_zone_pressure(board, w_king_sq, chess.WHITE)
        b_king_sq = board.king(chess.BLACK)
        if b_king_sq is not None:
            b_king_pressure = _king_zone_pressure(board, b_king_sq, chess.BLACK)

    my_pins = w_pins if board.turn == chess.WHITE else b_pins
    enemy_pins = b_pins if board.turn == chess.WHITE else w_pins
    
    my_pressure = w_king_pressure if board.turn == chess.WHITE else b_king_pressure
    enemy_pressure = b_king_pressure if board.turn == chess.WHITE else w_king_pressure

    score += (enemy_pins - my_pins)
    score += (enemy_pressure - my_pressure)

    return score


class Searcher:
    def __init__(self) -> None:
        self.tt: dict[int, tuple[int, int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [
            [None, None] for _ in range(MAX_PLY + 2)
        ]
        self.history: dict[tuple[chess.Color, int, int], int] = {}
        self.nodes = 0
        self.deadline = 0.0
        self.soft_deadline = 0.0
        self.path: dict[Hashable, int] = {}

    def _checkup(self) -> None:
        if time.monotonic() >= self.deadline:
            raise _Timeout

    def _has_non_pawn_material(self, board: chess.Board) -> bool:
        side = board.occupied_co[board.turn]
        return bool(side & (board.knights | board.bishops | board.rooks | board.queens))

    def _score_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        tt_move: chess.Move | None,
        ply: int,
    ) -> list[tuple[int, bool, int | None, chess.Move]]:
        killers = self.killers[ply]
        k0, k1 = killers[0], killers[1]
        history = self.history
        turn = board.turn

        scored: list[tuple[int, bool, int | None, chess.Move]] = []
        for m in moves:
            to_sq = m.to_square
            victim = board.piece_type_at(to_sq)
            is_ep = victim is None and board.is_en_passant(m)
            is_cap = victim is not None or is_ep

            if m == tt_move:
                key = TT_BONUS
            elif is_cap:
                attacker = board.piece_type_at(m.from_square) or 1
                v = SEE_VALUE[victim] if victim is not None else SEE_VALUE[None]
                key = CAPTURE_BONUS + v * 32 - SEE_VALUE[attacker]
                if m.promotion:
                    key += SEE_VALUE[m.promotion] * 16
            elif m.promotion:
                key = CAPTURE_BONUS + SEE_VALUE[m.promotion] * 16
            elif m == k0:
                key = KILLER1_BONUS
            elif m == k1:
                key = KILLER2_BONUS
            elif board.gives_check(m):
                key = CHECK_BONUS + history.get((turn, m.from_square, to_sq), 0)
            else:
                key = history.get((turn, m.from_square, to_sq), 0)

            scored.append((key, is_cap, victim, m))

        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def _store(
        self,
        key: int,
        idx: int,
        depth: int,
        flag: int,
        value: int,
        move: chess.Move | None,
        ply: int,
    ) -> None:
        if value > MATE_BOUND:
            value += ply
        elif value < -MATE_BOUND:
            value -= ply
        prev = self.tt.get(idx)
        if prev is None or prev[0] != key or prev[1] <= depth:
            self.tt[idx] = (key, depth, flag, value, move)

    def qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int, qply: int = 0) -> int:
        self.nodes += 1
        if not self.nodes & 255:
            self._checkup()
        if ply >= MAX_PLY:
            return evaluate(board)

        # Transposition Table lookup for Q-Search
        tkey = board._transposition_key()
        key = hash(tkey)
        idx = key % TT_MAX_ENTRIES
        entry = self.tt.get(idx)
        if entry is not None and entry[0] == key:
            _e_key, e_depth, e_flag, e_val, e_move = entry
            val = e_val
            if val > MATE_BOUND:
                val -= ply
            elif val < -MATE_BOUND:
                val += ply
            if e_flag == EXACT:
                return val
            if e_flag == LOWER and val >= beta:
                return val
            if e_flag == UPPER and val <= alpha:
                return val

        in_check = board.is_check()

        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            best = -INF
        else:
            stand = evaluate(board)
            if stand >= beta:
                self._store(key, idx, 0, LOWER, stand, None, ply)
                return stand
            if stand > alpha:
                alpha = stand
            best = stand
            
            # Generate captures, promotions, and limited checks
            moves = list(board.generate_legal_captures())
            for m in board.legal_moves:
                if not board.is_capture(m):
                    if m.promotion == 5:
                        moves.append(m)
                    elif qply < 2 and board.gives_check(m):
                        moves.append(m)

        scored = self._score_moves(board, moves, None, ply)

        for _k, _is_cap, victim, move in scored:
            if not in_check:
                gain = SEE_VALUE[victim] if victim is not None else 100
                if move.promotion:
                    gain += SEE_VALUE[move.promotion]
                if best + gain + 200 < alpha:
                    continue

            board.push(move)
            score = -self.qsearch(board, -beta, -alpha, ply + 1, qply + 1)
            board.pop()

            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        
        # Store Q-Search evaluation to avoid redundant processing
        flag = LOWER if best >= beta else (EXACT if best > alpha else UPPER)
        self._store(key, idx, 0, flag, best, None, ply)
        return best

    def search(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        can_null: bool = True,
    ) -> int:
        self.nodes += 1
        if not self.nodes & 255:
            self._checkup()

        if ply >= MAX_PLY:
            return evaluate(board)

        tkey = board._transposition_key()

        if ply > 0:
            if board.halfmove_clock >= 100:
                return 0
            if board.occupied.bit_count() <= 4 and board.is_insufficient_material():
                return 0
            if self.path.get(tkey, 0) >= 1:
                return 0

        in_check = board.is_check()
        if in_check:
            depth += 1  

        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply)

        alpha_orig = alpha
        key = hash(tkey)
        idx = key % TT_MAX_ENTRIES
        tt_move: chess.Move | None = None
        entry = self.tt.get(idx)
        if entry is not None and entry[0] == key:
            _e_key, e_depth, e_flag, e_val, e_move = entry
            tt_move = e_move
            if e_depth >= depth and ply > 0:
                val = e_val
                if val > MATE_BOUND:
                    val -= ply
                elif val < -MATE_BOUND:
                    val += ply
                if e_flag == EXACT:
                    return val
                if e_flag == LOWER and val >= beta:
                    return val
                if e_flag == UPPER and val <= alpha:
                    return val
            if tt_move is not None and not board.is_legal(tt_move):
                tt_move = None

        static: int | None = None

        if not in_check and abs(beta) < MATE_BOUND:
            static = evaluate(board)
            # Reverse Futility Pruning (Static evaluation is way above beta)
            if depth <= 4 and static - 85 * depth >= beta:
                return static

        if (
            can_null
            and not in_check
            and depth >= 3
            and ply > 0
            and abs(beta) < MATE_BOUND
            and self._has_non_pawn_material(board)
        ):
            if static is None:
                static = evaluate(board)
            if static >= beta:
                r = 2 + (depth // 4) + min(3, (static - beta) // 200)
                board.push(chess.Move.null())
                score = -self.search(board, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
                board.pop()
                if score >= beta:
                    if score > MATE_BOUND:
                        score = beta
                    return score

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0

        scored = self._score_moves(board, moves, tt_move, ply)

        best = -INF
        best_move: chess.Move | None = None
        key0 = tkey
        self.path[key0] = self.path.get(key0, 0) + 1
        
        futility_margin = 150 * depth

        try:
            for i, (_key, is_capture, _victim, move) in enumerate(scored):
                is_quiet = not is_capture and move.promotion is None
                gives_check = board.gives_check(move)

                # Futility Pruning (Skip hopeless quiet moves at low depth)
                if (
                    is_quiet
                    and not in_check
                    and not gives_check
                    and depth <= 2
                    and i > 0
                    and static is not None
                    and static + futility_margin <= alpha
                ):
                    continue

                # Late Move Pruning
                if (
                    is_quiet
                    and not in_check
                    and not gives_check
                    and depth <= 3
                    and i > 6 + depth * 6
                    and best > -MATE_BOUND
                ):
                    continue

                board.push(move)

                # Late Move Reduction
                reduction = 0
                if is_quiet and not gives_check and depth >= 3 and i >= 3:
                    reduction = int(0.5 + math.log(depth) * math.log(i + 1) / 2.2)
                    if reduction > depth - 2:
                        reduction = depth - 2
                    if reduction < 0:
                        reduction = 0

                if i == 0:
                    score = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self.search(
                        board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1
                    )
                    if score > alpha and reduction:
                        score = -self.search(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self.search(board, depth - 1, -beta, -alpha, ply + 1)

                board.pop()

                if score > best:
                    best = score
                    best_move = move
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            if is_quiet:
                                k = self.killers[ply]
                                if k[0] != move:
                                    k[1] = k[0]
                                    k[0] = move
                                hk = (board.turn, move.from_square, move.to_square)
                                self.history[hk] = self.history.get(hk, 0) + depth * depth
                            break
        finally:
            c = self.path.get(key0, 1) - 1
            if c <= 0:
                self.path.pop(key0, None)
            else:
                self.path[key0] = c

        if best >= beta:
            flag = LOWER
        elif best > alpha_orig:
            flag = EXACT
        else:
            flag = UPPER
        self._store(key, idx, depth, flag, best, best_move, ply)

        return best

    def go(
        self,
        board: chess.Board,
        hard_ms: float,
        soft_ms: float,
        max_depth: int = MAX_PLY,
        avoid: chess.Move | None = None,
    ) -> tuple[chess.Move | None, int, int]:
        now = time.monotonic()
        self.deadline = now + hard_ms / 1000.0
        self.soft_deadline = now + soft_ms / 1000.0
        self.nodes = 0

        if board.halfmove_clock == 0:
            self.tt.clear()

        if self.history:
            self.history = {k: v >> 3 for k, v in self.history.items() if v >= 8}

        self.path = dict(_GAME_HIST)

        moves = list(board.legal_moves)
        if not moves:
            return None, 0, 0
        if len(moves) == 1:
            return moves[0], 0, 1

        root_stack = len(board.move_stack)
        best_move: chess.Move | None = moves[0]
        best_score = 0
        depth_done = 0
        prev_scores: dict[chess.Move, int] = {}
        self.path[board._transposition_key()] = 1

        start = START_DEPTH if hard_ms >= START_DEPTH_MIN_MS else 1
        for depth in range(start, max_depth + 1):
            if depth > start and time.monotonic() >= self.soft_deadline:
                break

            moves.sort(
                key=lambda m: (
                    2 << 30 if m == best_move else prev_scores.get(m, -INF)
                ),
                reverse=True,
            )

            alpha, beta = -INF, INF
            if depth >= 5 and depth_done:
                window = 40
                alpha = best_score - window
                beta = best_score + window

            try:
                while True:
                    iter_best: chess.Move | None = None
                    iter_score = -INF
                    a = alpha
                    failed = False

                    for i, move in enumerate(moves):
                        board.push(move)
                        if i == 0:
                            score = -self.search(board, depth - 1, -beta, -a, 1)
                        else:
                            score = -self.search(board, depth - 1, -a - 1, -a, 1)
                            if a < score < beta:
                                score = -self.search(board, depth - 1, -beta, -a, 1)
                        board.pop()

                        if move == avoid:
                            score -= UNDO_PENALTY

                        prev_scores[move] = score

                        if score > iter_score:
                            iter_score = score
                            iter_best = move
                            if score > a:
                                a = score
                        if score >= beta:
                            failed = True
                            break

                    if failed:  
                        beta = min(INF, beta + 200 + (beta - alpha))
                        continue
                    if iter_score <= alpha and alpha != -INF:  
                        alpha = max(-INF, alpha - 200 - (beta - alpha))
                        beta = INF
                        continue
                    break

                best_move = iter_best
                best_score = iter_score
                depth_done = depth

                if abs(best_score) > MATE_BOUND:
                    break  

            except _Timeout:
                while len(board.move_stack) > root_stack:
                    board.pop()
                break

        while len(board.move_stack) > root_stack:
            board.pop()
        return best_move, best_score, depth_done

# Opening book integration
_BOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "book.bin")

def _open_book() -> chess.polyglot.MemoryMappedReader | None:
    try:
        if os.path.exists(_BOOK_PATH):
            reader = chess.polyglot.open_reader(_BOOK_PATH)
            print(f"agent: opening book loaded from {_BOOK_PATH}", file=sys.stderr)
            return reader
    except Exception as exc:
        print(f"agent: could not open book: {exc!r}", file=sys.stderr)
    return None

_BOOK_READER = _open_book()

def _book_move(board: chess.Board) -> chess.Move | None:
    if _BOOK_READER is None:
        return None
    try:
        entry = _BOOK_READER.weighted_choice(board)
        if entry and entry.move in board.legal_moves:
            return entry.move
    except Exception:
        pass
    return None

# Game state
_SEARCHER = Searcher()
_GAME_HIST: dict[Hashable, int] = {}
_LAST_PLY = -10
_LAST_OWN_MOVE: chess.Move | None = None

def _undo_move(board: chess.Board) -> chess.Move | None:
    last = _LAST_OWN_MOVE
    if last is None:
        return None
    try:
        mv = chess.Move(last.to_square, last.from_square)
        if not board.is_legal(mv) or board.is_capture(mv):
            return None
        if board.gives_check(mv):
            return None
        return mv
    except Exception:
        return None

def _note_position(board: chess.Board) -> None:
    try:
        if board.halfmove_clock == 0:
            _GAME_HIST.clear()
        key = board._transposition_key()
        _GAME_HIST[key] = _GAME_HIST.get(key, 0) + 1
    except Exception:
        pass

def _note_root(board: chess.Board) -> None:
    global _LAST_PLY, _LAST_OWN_MOVE
    try:
        ply = 2 * (board.fullmove_number - 1) + (0 if board.turn else 1)
        if ply != _LAST_PLY + 2:
            _GAME_HIST.clear()
            _LAST_OWN_MOVE = None
        _LAST_PLY = ply
    except Exception:
        _GAME_HIST.clear()
        _LAST_OWN_MOVE = None
    _note_position(board)


# --- Time Management ---

OVERHEAD_MS = 120
INCREMENT_MS = 1000      # Updated for 120+1 Lichess
PANIC_FLOOR_MS = 250
MOVES_REMAINING = 30.0   
HARD_MULTIPLIER = 2.6    
MAX_FRACTION = 0.30      

def _budget(time_left_ms: int) -> tuple[float, float]:
    left = time_left_ms - OVERHEAD_MS
    if left <= 0:
        return 1.0, 1.0
    if left < 1500:
        t = max(15.0, left * 0.12)
        return t, t * 0.5
    soft = left / MOVES_REMAINING + INCREMENT_MS * 0.70
    hard = soft * HARD_MULTIPLIER
    cap = left * MAX_FRACTION
    if hard > cap:
        hard = cap
    if soft > hard * 0.55:
        soft = hard * 0.55
    return hard, soft

def _safe_fallback(board: chess.Board) -> str:
    them = not board.turn
    best: chess.Move | None = None
    best_score = -1 << 30

    for move in board.legal_moves:
        victim = board.piece_type_at(move.to_square)
        gain = SEE_VALUE[victim] if victim is not None else 0
        mover = board.piece_type_at(move.from_square) or 1
        board.push(move)
        mate = board.is_checkmate()
        hanging = board.is_attacked_by(them, move.to_square)
        risk = 0 if mate else (SEE_VALUE[mover] if hanging else 0)
        board.pop()
        score = gain - risk + (1 << 20 if mate else 0)
        if score > best_score:
            best_score = score
            best = move
    return best.uci() if best is not None else "0000"

def get_move(fen: str, time_left_ms: int) -> str:
    fallback = "0000"
    try:
        board = chess.Board(fen)
    except Exception:
        return fallback

    try:
        if not any(board.legal_moves):
            return fallback
        fallback = _safe_fallback(board)
    except Exception as exc:
        print(f"agent: fallback selection failed: {exc!r}", file=sys.stderr)
        return fallback

    try:
        tl = int(time_left_ms)
    except Exception:
        tl = 1000

    _note_root(board)

    if tl <= PANIC_FLOOR_MS:
        _note_after(board, fallback)
        return fallback

    book_move = _book_move(board)
    if book_move is not None:
        chosen = book_move.uci()
        print(f"agent: playing book move {chosen}", file=sys.stderr)
        _note_after(board, chosen)
        return chosen

    chosen = fallback
    try:
        hard, soft = _budget(tl)
        move, _score, _depth = _SEARCHER.go(board, hard, soft, avoid=_undo_move(board))
        if move is not None and board.is_legal(move):
            chosen = move.uci()
    except Exception:
        traceback.print_exc(file=sys.stderr)

    _note_after(board, chosen)
    return chosen

def _note_after(board: chess.Board, uci: str) -> None:
    global _LAST_OWN_MOVE
    try:
        mv = chess.Move.from_uci(uci)
        board.push(mv)
        _note_position(board)
        board.pop()
        _LAST_OWN_MOVE = mv
    except Exception:
        _LAST_OWN_MOVE = None

# --- JIT WARMUP ---
print("agent: Warming up JIT compiler...", file=sys.stderr)
_dummy_board = chess.Board()
evaluate(_dummy_board)
print("agent: JIT compilation finished.", file=sys.stderr)