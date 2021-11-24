import argparse
from pathlib import Path

from apitep_utils import ArgumentParserHelper
from apitep_utils.analysis_modelling import AnalysisModeling
import logging
import numpy as np
import keys
from data_model.school_kind import SchoolKind
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly as py
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

log = logging.getLogger(__name__)


class QuadrimestersEnsemble(AnalysisModeling):
    quadrimester: int = None
    course: int = None
    school_kind: SchoolKind = None
    final_analys_record_personal_access: pd.DataFrame = None
    model_number: str = None
    x_train: pd.DataFrame = None
    x_test: pd.DataFrame = None
    x_train_norm: pd.DataFrame = None
    x_test_norm: pd.DataFrame = None
    y_train: pd.Series = None
    y_test: pd.Series = None
    y_pred: pd.Series = None
    threshold: float = 0.47

    def get_best_hyperparameters_RandomForest(self):
        grid_params = {'max_depth': [8, 12],
                       'n_estimators': [50, 100, 200]}
        gs_RndForest = GridSearchCV(
            RandomForestClassifier(random_state=123),
            grid_params,
            scoring='accuracy',
            n_jobs=-1,
            cv=4
        )
        gs_RndForest.fit(self.x_train, self.y_train)
        return gs_RndForest.best_estimator_

    def get_best_hyperparameters_SVM(self):
        grid_params = {'C': [0.01, 0.1, 1, 10, 50, 100, 200],
                       'gamma': [0.001, 0.01, 0.1, 1, 10]}

        gs_SVM = GridSearchCV(
            SVC(random_state=123),
            grid_params,
            scoring='accuracy',
            n_jobs=-1,
            cv=4
        )
        gs_SVM.fit(self.x_train_norm, self.y_train)
        return gs_SVM.best_estimator_

    @staticmethod
    def print_cv_results(cv_estimate):
        res = '\n Mean performance metric = %4.3f' % np.mean(cv_estimate) + '\n'
        res = res + ' SDT of the metric       = %4.3f' % np.std(cv_estimate) + '\n'
        res = res + ' Outcomes by cv fold' + '\n'
        for i, x in enumerate(cv_estimate):
            res = res + ' Fold %2d    %4.3f' % (i + 1, x) + '\n'
        return res

    @staticmethod
    def get_type_correlation(independent_feature: pd.Series, target_feature: pd.Series, positive_correlation: list,
                             negative_correlation: list):
        from scipy import stats
        corr = stats.spearmanr(independent_feature, target_feature).correlation
        if corr > 0:
            positive_correlation.append(independent_feature.name)
        if corr < 0:
            negative_correlation.append(independent_feature.name)

    def parse_arguments(self):
        """
        Parse arguments provided via command line, and check if they are valid
        or not. Adequate defaults are provided when possible.

        Parsed arguments are:
        - paths to the input CSV datasets, separated with spaces.
        - path to the output CSV dataset.
        """

        log.info("Get integration arguments")
        log.debug("QuadrimestersEnsemble.parse_arguments()")

        argument_parser = argparse.ArgumentParser(description=self.description)
        argument_parser.add_argument("-i", "--input_path", required=True,
                                     help="path to the input CSV dataset")
        argument_parser.add_argument("-o", "--output_path", required=True,
                                     help="path to the output CSV dataset")
        argument_parser.add_argument("-s", "--school_kind", required=True,
                                     help="school kind to analyze")
        argument_parser.add_argument("-c", "--course", required=True,
                                     help="course to analyze")
        argument_parser.add_argument("-q", "--quadrimester", required=True,
                                     help="quadrimester of course to analyze")
        argument_parser.add_argument("-mn", "--model_number", required=True,
                                     help="number of model to create")

        arguments = argument_parser.parse_args()
        self.input_path_segment = ArgumentParserHelper.parse_data_file_path(
            data_file_path=arguments.input_path)
        self.output_path_segment = ArgumentParserHelper.parse_data_file_path(
            data_file_path=arguments.output_path,
            check_is_file=False)
        self.course = int(arguments.course)
        self.quadrimester = int(arguments.quadrimester)
        self.temodel_number = arguments.model_number
        school_kind_str = arguments.school_kind
        if school_kind_str == "Teaching":
            self.school_kind = SchoolKind.Teaching
        elif school_kind_str == "Polytechnic":
            self.school_kind = SchoolKind.Polytechnic

    def process(self):
        """
        Analysis of final_analys_record_personal_access
        """
        if self.course == 0:
            log.info("Analysis of analys_record_personal_access data of school: " + self.school_kind.value)
        else:
            log.info("Analysis of analys_record_personal_access data of school: " + self.school_kind.value +
                     " with data of course " + str(self.course) + " and quadrimester " + str(self.quadrimester))

        log.debug("QuadrimestersEnsemble.process()")

        self.input_df.drop(['municipio'], axis=1)

        self.final_analys_record_personal_access = self.input_df.copy()

        self.input_df = pd.get_dummies(data=self.input_df,
                                       columns=self.input_df.drop(self.input_df.select_dtypes(
                                           include=['int64', 'float64']).columns, axis=1).columns)
        self.input_df.drop([keys.PLAN_CODE_KEY, keys.RECORD_KEY], axis=1, inplace=True)

    def analise(self):

        x_smt, y_smt = SMOTE(random_state=123).fit_sample(self.input_df.drop([keys.DROP_OUT_KEY], axis=1),
                                               self.input_df[keys.DROP_OUT_KEY])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_smt, y_smt, test_size=0.25,
                                                                                random_state=24, stratify=y_smt)

        self.models_developed.append(GradientBoostingClassifier(random_state=123).fit(self.x_train, self.y_train))

        y_pred = self.models_developed[0].predict(self.x_test)
        log.info("accuracy of GB model is: " + str(sklm.accuracy_score(y_true=self.y_test, y_pred=y_pred)))
        log.info("confusion matrix of GB model is: \n" + str(sklm.confusion_matrix(y_true=self.y_test, y_pred=y_pred)))
        log.info("recall of GB model is: " + str(sklm.recall_score(y_true=self.y_test, y_pred=y_pred)))

        x_test = self.input_df.drop([keys.DROP_OUT_KEY], axis=1)
        y_test = self.input_df[keys.DROP_OUT_KEY]

        y_pred = self.models_developed[0].predict(x_test)

        log.info("accuracy of model with complete data is: " + str(sklm.accuracy_score(y_true=y_test, y_pred=y_pred)))
        log.info("confusion matrix of model with complete data is: \n" + str(sklm.confusion_matrix(y_true=y_test,
                                                                                                   y_pred=y_pred)))
        log.info("recall of model with complete data is: " + str(sklm.recall_score(y_true=y_test, y_pred=y_pred)))

        self.y_pred = y_pred

    def save(self):

        output_path = Path(self.output_path_segment)
        output_path_parent = output_path.parent
        if not output_path_parent.exists():
            output_path_parent.mkdir(parents=True)

        self.final_analys_record_personal_access[self.model_number] = self.y_pred

        self.final_analys_record_personal_access.to_csv(
            self.output_path_segment,
            sep=self.output_separator,
            index=False)


def main():
    logging.basicConfig(
        filename="debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("--------------------------------------------------------------------------------------")
    log.info("Start QuadrimestersEnsemble")
    log.debug("main()")

    analys = QuadrimestersEnsemble(
        input_separator='|',
        output_separator='|'
    )
    analys.execute()


if __name__ == "__main__":
    main()
