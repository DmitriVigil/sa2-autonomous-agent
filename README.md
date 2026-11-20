# sa2-autonomous-agent
Modular deep learning framework using PyTorch and OpenCV to autonomously play Sonic Adventure 2 across Speed, Mech, and Treasure Hunting stages

## tech stack
```Python, PyTorch, OpenCV,mss and vgamepad ```

## folder structure 
``` sa2-autonomous-agent/
sa2-autonomous-agent/
├── core/
│   ├── capture.py          # mss + OpenCV screen grabbing & preprocessing
│   ├── controller.py       # Virtual gamepad (vgamepad) / keyboard output
│   └── recorder.py         # Universal gameplay recorder for any stage
├── models/
│   ├── speed_cnn.py        # Reactive CNN for Sonic & Shadow stages
│   ├── mech_cnn.py         # CNN with lock-on timing for Tails & Eggman
│   ├── hunting_lstm.py     # CNN + LSTM (memory) + radar parser for Knuckles & Rouge
│   └── stage_router.py     # Lightweight classifier to auto-switch models
|-stage_router.py 
├── train.py                # Unified PyTorch training loop
├── play.py                 # Live inference loop
└── requirements.txt
```