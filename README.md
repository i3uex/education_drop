# A Real-Life Machine Learning Experience for Predicting University Dropout at Different Stages Using Academic Data


High levels of school dropout are a major burden on the educational and professional development of a country’s inhabitants. A country’s prosperity depends, among other factors, on its ability to produce higher education graduates capable of moving a country forward. To alleviate the dropout problem, more and more institutions are turning to the possibilities that artificial intelligence can provide to predict dropout as early as possible. The difficulty of accessing personal data and privacy issues that it entails force the institutions to rely on the Academic Data of their students to create accurate and reliable predictive systems. This work focuses on creating the best possible predictive model based solely on academic data, and accordingly, its capacity to infer knowledge must be maximised. Thus, Feature Engineering and Instance Engineering techniques such as dealing with redundancy, significance of the features, correlation, cardinality features, missing values, creation or elimination of features, data fusion, removal of unuseful instances, binning, resampling, normalisation, or encoding are applied in detail before the construction of well-known models such as Gradient Boosting, Random Forest, and Support Vector Machine along with an Ensemble of them at different stages: prior to enrolment, at the end of the first semester, at the end of the second semester, at the end of the third semester, and at the end of the fourth semester. Through the construction of these predictive models that serve as inputs to a decision support system, the application of effective dropout prevention policies can be applied.

## Data
The original data from this study are not available because they are private data. However, in order that this study can be replicated, we indicate below the structure of the 3 main datasets used to carry out the study. If the user wants to reproduce it easily, he can generate a synthetic dataset with this structure.

__Enrolment data:__
| Feature                        | Description                                                                             | Class Description                                                                                     | # Class |
| ------------------------------ | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------- |
| Id                             | Student ID                                                                              | Numerical Values                                                                                      | 1718    |
| Degree ID                      | Unique Degree Identifier                                                                | Numerical Values                                                                                      | 16      |
| Degree Name                    | Degree Name                                                                             | Categorical: Computer Science Engineering, Civil Engineering, Telecommunication Engineering...        | 16      |
| Enrolment Year                 | First Academic Year of Studies                                                          | Categorical: 2007-08, 2008-09...                                                                      | 14      |
| Closed                         | Indicate whether the record is closed                                                   | Binary: Y, N                                                                                          | 2       |
| Transferred                    | Indicate whether the record is transferred to other institution                         | Binary: Y, N                                                                                          | 2       |
| TransferType                   | Indicates the reason for the transfer of a record                                      | Categorical: Internal, External, Simultaneous                                                          | 3       |
| Blocked                        | Indicate whether the record is blocked                                                  | Binary: Y, N                                                                                          | 2       |
| Call                           | Call for Access                                                                         | Categorical: June, September...                                                                       | 8       |
| Call Year                      | Year of the Call for Access                                                             | Categorical: 2007-08, 2008-09...                                                                      | 14      |
| Access ID                      | Access Type Unique Identifier                                                           | Numerical Value                                                                                       | –       |
| Access Description             | Access Type Description                                                                 | Categorical: University Entrance Exam, Transferred, 25 years of age or older, validated diploma...    | 11      |
| SubAccess ID                   | Subaccess Type Unique Identifier                                                        | Numerical Value                                                                                       | –       |
| SubAccess Description          | Subaccess Type Description                                                              | Categorical: LOE, LOGSE, LOMCE...                                                                     | 7       |
| Marks                          | University Entrance Exam                                                                | Numerical Value between 0 and 14                                                                       | –       |
| Origin Educational Institution | Origin Educational Institution                                                          | Categorical: List of names of the origin High Schools                                                 | 137     |
| Gender                         | Student's gender                                                                        | Categorical: H, M                                                                                     | 2       |
| Birth                          | Date of Birth                                                                           | Date                                                                                                  | –       |
| Province ID                    | Province ID                                                                             | Numerical Value                                                                                       | –       |
| Province Name                  | Province Name                                                                           | Categorical: Salamanca, Toledo, Sevilla, Alicante...                                                  | 50      |
| Municipality ID                | Municipality ID                                                                         | Numerical Value                                                                                       | –       |
| Municipality Name              | Municipality Name                                                                       | Categorical: Sevilla, Madrid, Salamanca, Toledo...                                                    | 434     |
| Dropout                        | Indicates if the student has dropped out                                                | Binary: Y, N                                                                                          | 2       |

__Qualifications data:__
| Feature                  | Description                                                       | Class Description                                                                                     | # Class |
| ------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------- |
| Id                       | Student ID                                                        | Numerical Values                                                                                      | 1718    |
| Degree ID                | Unique Degree Identifier                                          | Numerical Values                                                                                      | 16      |
| Degree Name              | Degree Name                                                       | Categorical: Computer Science Engineering, Civil Engineering, Telecommunication Engineering...        | 16      |
| Subject ID               | Unique Subject Identifier                                         | Numerical Values                                                                                      | –       |
| Subject Name             | Name of the Subject                                               | Categorical: Physics, Linear Algebra, Data Structures and Algorithms, Calculus, Economics and Business… | 332     |
| Year                     | Year                                                              | Numerical Values between 1 and 4                                                                       | –       |
| Semester                 | Semester of the Academic Year                                     | Categorical: 1S, 2S, Annual                                                                           | 3       |
| Subject Type ID          | Subject Type Unique Identifier                                    | Categorical: B, T, O, P, C, E                                                                         | 6       |
| Subject Type Description | Subject Type Description                                          | Categorical: Core, Compulsory, Optional, Internship…                                                   | 6       |
| Academic Year            | Academic Year                                                     | Categorical: 2007-08, 2008-09...                                                                      | 14      |
| Call                     | Call of Subject Examination                                       | Categorical: June, September, February...                                                             | 8       |
| Mark                     | Mark                                                              | Categorical: Not Taken, Fail, Compensation, Sufficient, Very Good, Outstanding, With Honours          | 7       |
| Numerical Mark           | Mark in Numerical Format                                          | Numerical Values from 0 to 10                                                                          | –       |
| Attempt                  | Attempt Number                                                    | Numerical Values from 1 to 6                                                                           | –       |

__Scolarship data:__

| Feature          | Description                                                      | Class Description                                            | # Class |
| ---------------- | ---------------------------------------------------------------- | ------------------------------------------------------------ | ------- |
| Id               | Student ID                                                       | Numerical Values                                             | 1718    |
| Academic Year    | Academic Year                                                    | Categorical: 2007-08, 2008-09...                             | 14      |
| Degree ID        | Unique Degree Identifier                                         | Numerical Values                                             | 16      |
| Degree Name      | Degree Name                                                      | Categorical: Computer Science Engineering, Civil Engineering, Telecommunication Engineering… | 16      |
| Scholarship      | Indicates if the student has a scholarship                       | Binary: Y, N                                                 | 2       |


## License
This project is licensed under the [MIT License](https://github.com/i3uex/education_drop/blob/main/LICENSE)

## Project structure:
This project is based on the framework defined by [MD4DSPRR Utils](https://github.com/i3uex/apitep_utils). 

This framework defines the structure of a data science project divided into the following phases:
- __ETL__: Extract Transform and Load data tasks
- __Feature Engineering__: process of creating or transforming features to enhance model performance.
- __Analysis and Modelling__: process of analysing data using machine learning models or other techniques.

Furthermore, in our case, using the ETL class, we will implement an additional phase called __Fetch Data__, which will allow us to give a common format to the raw data of the project.

Based on this framework, the structure of this project is as follows:
- __Data__:
    - __Raw__: raw originally data.
    - __Interim__: data with common format.
    - __Processed__: data resulting from the application of the ETL phase
    - __For_analyisis_and_modeling__: data resulting from the application of the Feature Engineering phase.
- __Code__:
  - __Ensemble__:
     - __Data_Acquisition_and_Understanding__: 
       - __Fetch_Data__: Data capture and formatting into a common format.
       - __ETL__:  Data that may provide noisy information to the models are eliminated, mainly due to being incomplete or not providing correct information.
       - __Analysis_and_Modeling__:
            - __Feature_Engineering__: features creation, elimination and fusion for personal access data and for course data.
            - __Feature_Selection__: features selection based on spearman correlation with target feature.
            - __Analysis__:
                 - ensemble_model.py: Ensemble model composed of Gradient Boosting, Random Forest, and SVM models.

## Prerequisites

- **Python 3.10** (virtual environment recommended)
- **Operating System:** Linux (or Windows Subsystem for Linux)
- **Dependencies:** All packages listed in `requirements.txt`

## Running the Pipeline

For run this project you have two options: 

1. Via Docker **(Recommended option for no worries by dependencies of Operative System)**:
   1. Install docker in your OS following these guidelines: https://docs.docker.com/get-docker/
   2. Make a pull of docker image by command "docker pull docker.pkg.github.com/i3uex/education_drop/education_drop_img:1.0"
   3. Run docker image by command "docker run docker.pkg.github.com/i3uex/education_drop/education_drop_img:1.0"

2. On premise **(Linux or Windows Subsystem for Linux and Python 3.8)**: 
   1. Go to the root directory of project and install all requirements by command "pip3 install -r requirements.txt"
   2. Execute the command "sh run_project.sh"

This will sequentially perform:
1. **fetch data**
2. **ETL**
3. **feature engineering**
4. **feature selection**
5. **analyis and modeling by ensemble model**

## Other branches:
- __exp/caise_knime_env__: Branch corresponding to the illustrative example of the article [A Model-Driven Approach for Systematic
Reproducibility and Replicability of Data
Science Projects](https://link.springer.com/chapter/10.1007/978-3-031-07472-1_9), associated with the "knime_env" environment.

- __exp/caise_python_env__: Branch corresponding to the illustrative example of the article [A Model-Driven Approach for Systematic
Reproducibility and Replicability of Data
Science Projects](https://link.springer.com/chapter/10.1007/978-3-031-07472-1_9), associated with the "python_env" environment.

- __exp/dv__: Branch for a specific data visualization of this project.
