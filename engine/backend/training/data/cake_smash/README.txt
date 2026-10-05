DROP CLIPS / IMAGES HERE TO TRAIN THE CAKE-SMASH DETECTOR
=========================================================
smash/      → the smash itself: face going INTO the cake (head down, face hidden)
              AND the aftermath (cream on the face, face visible). BOTH count.
not_smash/  → hard negatives: cake_cutting (knife), candle_blowing, cake_feeding
              (fork to mouth), plain gathering around the cake, people with cake.

Aim for >=20 clips per folder, varied people / angles / lighting / cake types.
Stills (.jpg/.png) work too. Then run:
    .venv/bin/python3 backend/training/train_cake_smash.py
