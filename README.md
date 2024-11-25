<h1 align="center" id="title">hierarchical autonomous racing with pit stops</h1>

<p align="center"><img src="https://socialify.git.ci/ghssx19/RL-Project-Gym/image?font=Inter&language=1&name=1&stargazers=1&theme=Auto" alt="project-image"></p>

<p id="description">Environment and base models for training and deploying multi or single agent races for hierarchical RL approaches.</p>

<h2>🚀 Demo</h2>

<h2>🧐 Status</h2>

<p>We are currently working on transferring over the project to IsaacLab</p>

<h2>🎥 Video Demos</h2>

<h3 align="center">Agent Racing in Real time<h3>
<p align="center"><img src="agent.gif" alt="demo-gif1"></p>

<h3 align="center">Human Racing in Real time<h3>
<p align="center"><img src="human.gif" alt="demo-gif2"></p>

<h2>📐 Architecture Diagram</h2>

<p align="center"><img src="YOUR_ARCHITECTURE_IMAGE_LINK" alt="architecture-diagram"></p>

<h2 align="center">🛠️ Installation Steps:</h2>

<p>1. 🔧 Clone the Repository</p>

<pre>
git clone https://github.com/ghssx19/RL-Project-Gym.git
</pre>

<p>2. Directory</p>

<pre>
cd multi_car_racing
</pre>

<p>3. Conda</p>

<pre>
conda env create -f dependencies/explicit-env-conda.txt
</pre>

<p>4. Pip</p>

<pre>
pip install -r dependencies/requirements_pip.txt
</pre>

<h2>▶️ Running the Code</h2>

<p>To see the main simulation where a single agent is performing pit stops, run the following command:</p>

<pre>
python main/main.py
</pre>

<p>This will launch the simulation environment where you can observe the agent's behavior and pit stop strategy in action.</p>

<h2 align="center">📖 Code Explanation</h2>

<p>The <code>multi_car_racing.py</code> file is used to create the environment for hierarchical RL racing. It supports both single agent and multi-agent configurations. Originally developed by <a href="http://www.www.hhp">www.www.hhp</a>, our team has made modifications and added wrappers to support hierarchical strategies. The <code>MultiAgentRacingLSTM.py</code> file extends our framework to multi-agent racing, which has highlighted issues like agent collisions leading to instability, as the lower-level model requires fine-tuning to penalize collisions. The <code>BenchmarkingAgainstHuman.py</code> file allows users to manually race against the environment while timing their laps. Lastly, <code>train.py</code> is used for training the models.</p>

## Acknowledgements

We would like to extend our thanks to the individuals and communities who made this project possible:

- **[igilitschenski & tseyde ](https://github.com/igilitschenski/multi_car_racing.git)** 🧑‍💻 - For their contributions to the environment used a starting point here.
- **[sb3](https://huggingface.co/sb3/ppo_lstm-CarRacing-v0)** 🤗 - For their contributions to the low-level model used as a starting point here.

Special thanks to the developers and maintainers of the libraries, frameworks, and tools that powered this project.
