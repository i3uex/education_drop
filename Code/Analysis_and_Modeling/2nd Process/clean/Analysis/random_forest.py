import argparse
from pathlib import Path

from apitep_utils import ArgumentParserHelper
from apitep_utils.analysis_modelling import AnalysisModeling
import logging
import numpy as np
import keys
from data_model.school_kind import SchoolKind
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as sklm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly as py

log = logging.getLogger(__name__)


class QuadrimestersRandomForest(AnalysisModeling):
    quadrimester: int = None
    course: int = None
    school_kind: SchoolKind = None
    final_analys_record_personal_access: pd.DataFrame = None
    model_number: str = None
    x_train: pd.DataFrame = None
    x_test: pd.DataFrame = None
    y_train: pd.Series = None
    y_test: pd.Series = None

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
        log.debug("Integration.parse_arguments()")

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
        self.model_number = arguments.model_number
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
                     "with data of course " + str(self.course) + " and quadrimester " + str(self.quadrimester))

        log.debug("QuadrimesterRandomForest.process()")

        if keys.CUM_MEDIAN_KEY in self.input_df.columns:
            median_bcket_array = np.array([0, 1.5, 3, 4.5, 6, 7.5, 9, 10])
            self.input_df[keys.CUM_MEDIAN_INTERVAL_KEY] = pd.cut(
                self.input_df[keys.CUM_MEDIAN_KEY], median_bcket_array, include_lowest=True)
            self.input_df.drop([keys.CUM_MEDIAN_KEY], axis=1, inplace=True)

        self.final_analys_record_personal_access = self.input_df.copy()
        self.input_df = pd.get_dummies(data=self.input_df,
                                       columns=self.input_df.drop(self.input_df.select_dtypes(
                                           include=['int64', 'float64']).columns, axis=1).columns)
        self.input_df.drop([keys.PLAN_CODE_KEY, keys.RECORD_KEY], axis=1, inplace=True)

    def analise(self):
        drop_out_data = self.input_df[self.input_df[keys.DROP_OUT_KEY] == 1]
        no_drop_out_data = self.input_df[self.input_df[keys.DROP_OUT_KEY] == 0]

        if drop_out_data.shape[0] > no_drop_out_data.shape[0]:
            from sklearn.utils import resample
            drop_out_data_downsampled = resample(drop_out_data,
                                                 replace=False,
                                                 n_samples=no_drop_out_data.shape[0],
                                                 random_state=123)
            resample_df = pd.concat([drop_out_data_downsampled, no_drop_out_data])
            x = resample_df.drop([keys.DROP_OUT_KEY], axis=1)
            y = resample_df[keys.DROP_OUT_KEY]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25,
                                                                                    random_state=24)
        elif self.course > 0:
            from imblearn.combine import SMOTETomek
            x_smt, y_smt = SMOTETomek().fit_sample(self.input_df.drop([keys.DROP_OUT_KEY], axis=1),
                                                   self.input_df[keys.DROP_OUT_KEY])

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_smt, y_smt, test_size=0.25,
                                                                                    random_state=24)
        else:
            from sklearn.utils import resample
            no_drop_out_data_downsampled = resample(no_drop_out_data,
                                                    replace=False,
                                                    n_samples=1500,
                                                    random_state=123)
            drop_out_data_upsampled = resample(no_drop_out_data,
                                               replace=True,
                                               n_samples=1500,
                                               random_state=123)
            resample_df = pd.concat([drop_out_data_upsampled, no_drop_out_data_downsampled])
            x = resample_df.drop([keys.DROP_OUT_KEY], axis=1)
            y = resample_df[keys.DROP_OUT_KEY]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25,
                                                                                    random_state=24)

        self.model_developed = GradientBoostingClassifier(random_state=123).fit(self.x_train, self.y_train)

    def save(self):
        y_pred = self.model_developed.predict(self.x_test)

        log.info("accuracy of model is: " + str(sklm.accuracy_score(y_true=self.y_test, y_pred=y_pred)))
        log.info("confusion matrix of model is: \n" + str(sklm.confusion_matrix(y_true=self.y_test, y_pred=y_pred)))
        log.info("recall of model is: " + str(sklm.recall_score(y_true=self.y_test, y_pred=y_pred)))

        log.info("recall of model applying cross validation algorithm is: " + str(self.print_cv_results(
            cross_val_score(self.model_developed, self.x_train, self.y_train, scoring='recall'))))

        x_test = self.input_df.drop([keys.DROP_OUT_KEY], axis=1)
        y_test = self.input_df[keys.DROP_OUT_KEY]

        y_pred = self.model_developed.predict(x_test)

        log.info("accuracy of model with complete data is: " + str(sklm.accuracy_score(y_true=y_test, y_pred=y_pred)))
        log.info("confusion matrix of model with complete data is: \n" + str(sklm.confusion_matrix(y_true=y_test,
                                                                                                   y_pred=y_pred)))
        log.info("recall of model with complete data is: " + str(sklm.recall_score(y_true=y_test, y_pred=y_pred)))

        feature_names = x_test.columns
        result = permutation_importance(
            self.model_developed, x_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)
        logit_importances = pd.Series(result.importances_mean, index=feature_names)
        logit_importances = logit_importances[logit_importances > 0.001]

        logit_importances = pd.DataFrame({'Feature': logit_importances.sort_values(ascending=False).index,
                                          'Permutation_importance': logit_importances.sort_values(
                                              ascending=False)}).reset_index(drop=True)

        fig = px.bar(logit_importances, x='Feature', y='Permutation_importance')
        output_path = Path(self.output_path_segment)
        output_path_parent = output_path.parent
        if not output_path_parent.exists():
            output_path_parent.mkdir(parents=True)

        path_segment = str(output_path_parent)
        output_path_plot = Path(path_segment) / 'random_forest_feature_importances'
        output_path_plot = output_path_plot.with_suffix(".html")
        py.offline.plot(fig, filename=str(output_path_plot))

        positive_correlation = []
        negative_correlation = []
        for feature in logit_importances['Feature']:
            self.get_type_correlation(self.input_df[feature],
                                      self.input_df[keys.DROP_OUT_KEY], positive_correlation,
                                      negative_correlation)

        log.info("features with positive correlation with target feature are: \n" + str(positive_correlation))
        log.info("features with negative correlation with target feature are: \n" + str(negative_correlation))

        self.final_analys_record_personal_access[self.model_number] = y_pred

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
    log.info("Start QuadrimestersRandomForest")
    log.debug("main()")

    analys = QuadrimestersRandomForest(
        input_separator='|',
        output_separator='|'
    )
    analys.execute()


if __name__ == "__main__":
    main()
