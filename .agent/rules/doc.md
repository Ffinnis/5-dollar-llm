---
trigger: always_on
---

Always read doc (All md files) before implement some changes at `docs` folder and README.md

Read all project data to understand current implementation before do something

After the implementation of the new experiment, I will test it on a separate instance to run a benchmark, so I can manually change params of the model, while testing

ALl the train_llm are happening on a separate instance, so you shouldn't run it on my pc

Always use .venv for running something

We're trying to set best possible speed on leaderboard with architecture updates, so the main goal is to optimize pretraining process as good as possblie without losing val_loss.