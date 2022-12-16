# LNS-PBS
Python implementation of "Multi-Goal Multi-Agent Pickup and Delivery [1]"

## Usage
Run main.sh
- 1. main_lns.py: **LNS**
     - Generates random scenarios
     - Initial agents-tasks assignment by hungarian method
     - Operates LNS procedure for several times
     - Generates an high-level task allocation
     
- 2. imports low-level path finding algorithm: **EECBS**
     - All copyrights of EECBS are in the possession of https://github.com/Jiaoyang-Li/EECBS
     - Create submodule /EECBS under dir /LNS-PBS by running ```git submodule add https://github.com/Jiaoyang-Li/EECBS.git EECBS```
     
- 3. main_eecbs.py
     
## References
[1] Xu, Qinghong, et al. "Multi-Goal Multi-Agent Pickup and Delivery." arXiv preprint arXiv:2208.01223 (2022).
