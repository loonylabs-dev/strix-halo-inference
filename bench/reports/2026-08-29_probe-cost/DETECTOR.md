# Would a passive degeneracy check work?

29.08.2026. No GPU, no server: the ground truth is already in this repo. Every
slot-corruption run recorded its answers with a per-answer verdict — 316
genuinely corrupted responses and 383 healthy ones, from this machine.

    python3 bench/suites/passive-degeneracy.py

## The rule as it stands, and two variants

    Variante                        gefunden  Fehlalarm  fällt herein auf
    heute:      min 24, 60 %          100.0%      0.0%   Trennlinie in Prosa, reine
                                                         Trennlinie, ASCII-Tabelle,
                                                         base64, Fortschrittsbalken
    strenger:   min 24, 90 %          100.0%      0.0%   reine Trennlinie, base64,
                                                         Fortschrittsbalken
    + nur Symbole (kein a-z0-9)       100.0%      0.0%   reine Trennlinie,
                                                         Fortschrittsbalken

All three catch every one of the 316. The difference is what else they catch.

## The finding that decides the design

The corrupted answers are SHORT: median 120 characters. A full-width markdown
rule is 80. So the obvious safety measure — only judge long answers — does not
work:

    min 200, 90 %      0.0 % gefunden
    min 400, 95 %      0.0 % gefunden

Raising the length threshold past the fault's own size removes the fault from
view. The discrimination has to come from the SHAPE, not the length.

Two shapes do it, and both keep 100 % recall:

* a share of 90 % rather than 60 % — a rule inside prose is 71 % and drops out;
* the dominant character must not be alphanumeric — which removes base64
  without touching `////`.

What survives: an answer that is ENTIRELY a rule, or entirely a progress bar.
Implausible as a whole model answer, not impossible.

## And the streaks close the gap

Corruption did not come single-file. Of 79 runs with any corrupted answer, the
longest chain per run had a **median of 4 and a minimum of 4** consecutive
corrupted responses. So requiring TWO in a row before acting costs one answer
of delay and removes the remaining false-alarm shapes, which do not repeat.

## What this supports

    passive   share >= 0.90, dominant character not alphanumeric,
              two consecutive answers      — costs nothing, needs no slot
    active    the arithmetic question ("391"), kept for CONFIRMATION and for
              idle machines, where the slot is free and the eviction that
              costs five minutes cannot happen

## What it does not support

Recall is measured against one signature: every corrupted sample in this
corpus is the `////` shape. A future corruption that looks different would be
missed by all three variants, and this measurement says nothing about it.
