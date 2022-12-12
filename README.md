# LNS-PBS
Python implementation of "Multi-Goal Multi-Agent Pickup and Delivery [1]"

## Usage
1. Create a random scenarios via generate_scenarios.py

2. Execute main.py
* Initial agents-tasks assignment by hungarian method
* Operates LNS procedure several times
* Generates an higher-level task allocation

3. Run main_eecbs.py
* EECBS is released under USC – Research License
* All copyrights are in the possession of https://github.com/Jiaoyang-Li/EECBS
* Operates EECBS lower-level path finding

## References
[1] Xu, Qinghong, et al. "Multi-Goal Multi-Agent Pickup and Delivery." arXiv preprint arXiv:2208.01223 (2022).
