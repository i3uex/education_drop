# Education drop

In this repository we have two process:
1. The process responsible of predicting school dropout
2. The process responsible of generate data necessary for data data display panel
3. The process for validation section of CAISE '22 Conference.

For run this process you have two options:
- Process responsible of predicting school dropout
1. Via Docker **(Recommended option for no worries by dependencies of Operative System)**:
   1. Install docker in your OS following these guidelines: https://docs.docker.com/get-docker/
   2. Make a pull of docker image by command "docker pull docker.pkg.github.com/i3uex/education_drop/ed_exp_ensemble_47:1.1"
   3. Run docker image by command "docker run docker.pkg.github.com/i3uex/education_drop/ed_exp_ensemble_47:1.1"

2. On premise **(Required Ubuntu OS 20.04 and Python 3.8)**: 
   1. Go to the root directory of branch exp/ensemble_47 and install all requirements by command "pip3 install -r requirements.txt"
   2. Execute the command "sh run_project.sh"

- Process responsible of generate data necessary for data data display panel
1. Via Docker **(Recommended option for no worries by dependencies of Operative System)**:
   1. Install docker in your OS following these guidelines: https://docs.docker.com/get-docker/
   2. Make a pull of docker image by command "docker pull docker.pkg.github.com/i3uex/education_drop/ed_exp_dv:1.1"
   3. Run docker image by command "docker run docker.pkg.github.com/i3uex/education_drop/ed_exp_dv:1.1"

2. On premise **(Required Ubuntu OS 20.04 and Python 3.8)**: 
   1. Go to the root directory of branch exp/dv and install all requirements by command "pip3 install -r requirements.txt"
   2. Execute the command "sh run_project.sh"
  
- Process for validation section of CAISE '22 Conference
1. Via Docker **(Recommended option for no worries by dependencies of Operative System)**:
   1. Install docker in your OS following these guidelines: https://docs.docker.com/get-docker/
   2. Make a pull of docker image by command "docker pull docker.pkg.github.com/i3uex/education_drop/caise_validation:v1.0"
   3. Run docker image by command "docker run docker.pkg.github.com/i3uex/education_drop/caise_validation:v1.0"

2. On premise **(Required Ubuntu OS 20.04 and Python 3.8)**: 
   1. Go to the root directory of branch exp/dv and install all requirements by command "pip3 install -r requirements.txt"
   2. Execute the command "sh run_project.sh"
