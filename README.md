# Education drop

In this repository we have three process:
1. The process responsible of predicting school dropout
2. The process responsible of generate data necessary for data data display panel
3. The process for **Illustrative expample** section of **CAISE '22 Conference**.

For run this process you have two options:
- **Process responsible of predicting school dropout**
1. Via Docker **(Recommended option for no worries by dependencies of Operative System)**:
   1. Install docker in your OS following these guidelines: https://docs.docker.com/get-docker/
   2. Make a pull of docker image by command "docker pull docker.pkg.github.com/i3uex/education_drop/ed_exp_ensemble_47:1.1"
   3. Run docker image by command "docker run docker.pkg.github.com/i3uex/education_drop/ed_exp_ensemble_47:1.1"

2. On premise **(Required Ubuntu OS 20.04 and Python 3.8)**: 
   1. Go to the root directory of branch exp/ensemble_47 and install all requirements by command "pip3 install -r requirements.txt"
   2. Execute the command "sh run_project.sh"

- **Process responsible of generate data necessary for data data display panel**
1. Via Docker **(Recommended option for no worries by dependencies of Operative System)**:
   1. Install docker in your OS following these guidelines: https://docs.docker.com/get-docker/
   2. Make a pull of docker image by command "docker pull docker.pkg.github.com/i3uex/education_drop/ed_exp_dv:1.1"
   3. Run docker image by command "docker run docker.pkg.github.com/i3uex/education_drop/ed_exp_dv:1.1"

2. On premise **(Required Ubuntu OS 20.04 and Python 3.8)**: 
   1. Go to the root directory of branch exp/dv and install all requirements by command "pip3 install -r requirements.txt"
   2. Execute the command "sh run_project.sh"
  
- **Process for validation section of CAISE '22 Conference**
1. **Python Environment** (The code is available on exp/caise_python_env)
   1. Install docker in your OS following these guidelines: https://docs.docker.com/get-docker/
   2. Make a pull of docker image by command "docker pull docker.pkg.github.com/i3uex/education_drop/caise_python_env:v1.0"
   3. Run docker image by command "docker run docker.pkg.github.com/i3uex/education_drop/caise_python_env:v1.0"

2. **KNIME Environment** (The code is available on exp/caise_knime_env): 
   1. Go to exp/caise_knime_env branch and follow "education_drop/Container/Vagrant" and read the instructions in the readme for install the requirements for execute the project.
   2. Once the execution environment and the requirements have been installed, in order to execute the defined pipeline, it is necessary to open the virtual environment created with Vagrant, run the KNIME tool that has been installed in the previous step and once in it, in the upper bar go to "File" >> "Import KNIME Workflow ...", and select the file "Analysis&Modeling.knwf" located in the "Code" folder of this branch and execute the imported pipeline through the F7 key.
