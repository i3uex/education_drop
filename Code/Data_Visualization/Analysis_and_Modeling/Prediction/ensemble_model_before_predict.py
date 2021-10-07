import argparse
from pathlib import Path
from typing import List

from apitep_utils import ArgumentParserHelper
from apitep_utils.analysis_modelling import AnalysisModeling
import logging
import numpy as np
import keys
from data_model.school_kind import SchoolKind
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as sklm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import pickle

log = logging.getLogger(__name__)


class QuadrimestersEnsemble(AnalysisModeling):
    quadrimester: int = None
    course: int = None
    school_kind: SchoolKind = None
    final_analys_record_personal_access: pd.DataFrame = None
    x_train: pd.DataFrame = None
    x_test: pd.DataFrame = None
    x_train_norm: pd.DataFrame = None
    x_test_norm: pd.DataFrame = None
    y_train: pd.Series = None
    y_test: pd.Series = None
    y_pred: pd.Series = None
    threshold: float = 0.47
    input_dfs: List = []
    input_path_segments: List = None
    train_ids: pd.DataFrame = None

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
        argument_parser.add_argument("-i", "--input_paths",
                                     required=True,
                                     nargs="+",
                                     help="path to the input CSV dataset")
        argument_parser.add_argument("-o", "--output_path", required=True,
                                     help="path to the output CSV dataset")
        argument_parser.add_argument("-s", "--school_kind", required=True,
                                     help="school kind to analyze")
        argument_parser.add_argument("-c", "--course", required=True,
                                     help="course to analyze")
        argument_parser.add_argument("-q", "--quadrimester", required=True,
                                     help="quadrimester of course to analyze")

        arguments = argument_parser.parse_args()
        input_path_segments = arguments.input_paths
        self.input_path_segments = []
        for input_path_segment in input_path_segments:
            self.input_path_segments.append(
                ArgumentParserHelper.parse_data_file_path(
                    data_file_path=input_path_segment)
            )
        self.output_path_segment = ArgumentParserHelper.parse_data_file_path(
            data_file_path=arguments.output_path,
            check_is_file=False)
        self.course = int(arguments.course)
        self.quadrimester = int(arguments.quadrimester)
        school_kind_str = arguments.school_kind
        if school_kind_str == "Teaching":
            self.school_kind = SchoolKind.Teaching
        elif school_kind_str == "Polytechnic":
            self.school_kind = SchoolKind.Polytechnic

    def load(self):
        """
        Load the CSV datasets in the input path list provided. Optionally, save
        a report in the same path for each of them, with the same file name, but
        with HTML extension.
        """

        log.info("Load input datasets")
        log.debug("Integration.load()")

        if self.input_path_segments is None:
            log.debug("- input path is none, nothing to load or report about")
            return
        if not self.input_type_excel:
            for input_path_segment in self.input_path_segments:
                input_df = pd.read_csv(
                    input_path_segment,
                    sep=self.input_separator)
                self.input_dfs.append(input_df)
        else:
            for input_path_segment in self.input_path_segments:
                input_df = pd.concat(pd.read_excel(
                    input_path_segment,
                    header=0,
                    sheet_name=None
                ), ignore_index=True)
                self.input_dfs.append(input_df)

        if self.save_report_on_load:
            for index, input_df in enumerate(self.input_dfs):
                input_path_segment = self.input_path_segments[index]
                self.save_report(input_df, input_path_segment)

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

        if keys.CUM_MEDIAN_KEY in self.input_dfs[0].columns:
            median_bcket_array = np.array([0, 1.5, 3, 4.5, 6, 7.5, 9, 10])
            self.input_dfs[0][keys.CUM_MEDIAN_INTERVAL_KEY] = pd.cut(
                self.input_dfs[0][keys.CUM_MEDIAN_KEY], median_bcket_array, include_lowest=True)
            self.input_dfs[0].drop([keys.CUM_MEDIAN_KEY], axis=1, inplace=True)

        # TRATAMIENTO DEL DATASET A PREDECIR

        final_colums = []
        if self.school_kind is SchoolKind.Polytechnic:
            if self.course < 1:
                final_colums = ['des_plan', 'anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'sexo', 'municipio', 'anio_nacimiento_interval', 'nota_admision_def']
            elif self.course == 1 and self.quadrimester == 1:
                final_colums = ['des_plan', 'anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'sexo', 'municipio', 'anio_nacimiento_interval', '1st_model', 'scholarship',
                                'cum_absent_ratio', 'nota_admision_def', 'cum_pass_ratio']
            elif self.course == 1 and self.quadrimester == 2:
                final_colums = ['des_plan', 'anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'sexo', 'municipio', 'anio_nacimiento_interval', '1st_model', 'scholarship',
                                '2nd_model', 'cum_pass_ratio', 'cum_absent_ratio', 'nota_admision_def']
            elif self.course == 2 and self.quadrimester == 1:
                final_colums = ['des_plan', 'anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'sexo', 'municipio', 'anio_nacimiento_interval', '1st_model', 'scholarship',
                                '2nd_model', '3rd_model', 'nota_admision_def', 'cum_pass_ratio', 'cum_median',
                                'cum_absent_ratio']
            elif self.course == 2 and self.quadrimester == 2:
                final_colums = ['des_plan', 'anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'sexo', 'municipio', 'anio_nacimiento_interval', '1st_model', 'scholarship',
                                '2nd_model', '3rd_model', '4th_model', 'nota_admision_def', 'cum_pass_ratio',
                                'cum_median', 'cum_absent_ratio']
            elif self.course == 3 and self.quadrimester == 1:
                final_colums = ['des_plan', 'anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'anio_nacimiento_interval', '1st_model', 'scholarship', '2nd_model', '3rd_model',
                                'cum_median', '4th_model', '5th_model', 'cum_absent_ratio', 'cum_pass_ratio',
                                'cum_more_1st_call_ratio', 'nota_admision_def']
            elif self.course == 3 and self.quadrimester == 2:
                final_colums = ['des_plan', 'anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'anio_nacimiento_interval', '1st_model', 'scholarship', '2nd_model', '3rd_model',
                                'cum_median', '4th_model', '5th_model', '6th_model', 'cum_absent_ratio',
                                'cum_more_1st_call_ratio', 'cum_pass_ratio', 'nota_admision_def']
            elif self.course == 4 and self.quadrimester == 1:
                final_colums = ['anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'anio_nacimiento_interval', '1st_model', 'scholarship', '2nd_model', '3rd_model',
                                'cum_median', '4th_model', '5th_model', '6th_model', '7th_model',
                                'cum_absent_ratio', 'cum_pass_ratio', 'nota_admision_def', 'cum_more_1st_call_ratio']
            elif self.course == 4 and self.quadrimester == 2:
                final_colums = ['anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'anio_nacimiento_interval', '1st_model', 'scholarship', '2nd_model', '3rd_model',
                                'cum_median', '4th_model', '5th_model', '6th_model', '7th_model', '8th_model',
                                'cum_more_1st_call_ratio', 'cum_pass_ratio', 'cum_absent_ratio',
                                'nota_admision_def']
            else:
                raise NotImplementedError

        final_colums.append(keys.RECORD_KEY)
        final_colums.append(keys.PLAN_CODE_KEY)
        self.input_dfs[1] = self.input_dfs[1][final_colums]

        if keys.CUM_MEDIAN_KEY in self.input_dfs[1].columns:
            median_bcket_array = np.array([0, 1.5, 3, 4.5, 6, 7.5, 9, 10])
            self.input_dfs[1][keys.CUM_MEDIAN_INTERVAL_KEY] = pd.cut(
                self.input_dfs[1][keys.CUM_MEDIAN_KEY], median_bcket_array, include_lowest=True)
            self.input_dfs[1].drop([keys.CUM_MEDIAN_KEY], axis=1, inplace=True)

        self.final_analys_record_personal_access = self.input_dfs[0].copy()

        # CONCATENACIÓN CON DATASET A PREDECIR

        self.input_dfs[0][keys.PLAN_CODE_RECORD_KEY] = self.input_dfs[0].apply(
            lambda func: int(str(func[keys.PLAN_CODE_KEY]) + str(func[keys.RECORD_KEY])), axis=1)
        self.input_dfs[1][keys.PLAN_CODE_RECORD_KEY] = self.input_dfs[1].apply(
            lambda func: int(str(func[keys.PLAN_CODE_KEY]) + str(func[keys.RECORD_KEY])), axis=1)
        valid_rows = self.input_dfs[0][keys.PLAN_CODE_RECORD_KEY]

        self.input_dfs[0] = pd.concat([self.input_dfs[0], self.input_dfs[1]], ignore_index=True)
        self.input_dfs[0].drop_duplicates(subset=[keys.PLAN_CODE_RECORD_KEY], keep='first', inplace=True)
        self.input_dfs[0] = pd.get_dummies(data=self.input_dfs[0],
                                           columns=self.input_dfs[0].drop(self.input_dfs[0].select_dtypes(
                                               include=['int64', 'float64']).columns, axis=1).columns)
        self.input_dfs[0] = self.input_dfs[0][self.input_dfs[0][keys.PLAN_CODE_RECORD_KEY].isin(valid_rows)]

    def analise(self):
        drop_out_data = self.input_dfs[0][self.input_dfs[0][keys.DROP_OUT_KEY] == 1]
        no_drop_out_data = self.input_dfs[0][self.input_dfs[0][keys.DROP_OUT_KEY] == 0]

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
            x_smt, y_smt = SMOTETomek().fit_sample(self.input_dfs[0].drop([keys.DROP_OUT_KEY], axis=1),
                                                   self.input_dfs[0][keys.DROP_OUT_KEY])

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_smt, y_smt, test_size=0.25,
                                                                                    random_state=24)
        else:
            from sklearn.utils import resample
            no_drop_out_data_downsampled = resample(no_drop_out_data,
                                                    replace=False,
                                                    n_samples=1500,
                                                    random_state=123)
            drop_out_data_upsampled = resample(drop_out_data,
                                               replace=True,
                                               n_samples=1500,
                                               random_state=123)
            resample_df = pd.concat([drop_out_data_upsampled, no_drop_out_data_downsampled])
            x = resample_df.drop([keys.DROP_OUT_KEY], axis=1)
            y = resample_df[keys.DROP_OUT_KEY]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25,
                                                                                    random_state=24)

        self.train_ids = pd.DataFrame()
        self.train_ids[keys.PLAN_CODE_RECORD_KEY] = self.x_train[keys.PLAN_CODE_RECORD_KEY]
        self.x_train.drop([keys.PLAN_CODE_KEY, keys.RECORD_KEY, keys.PLAN_CODE_RECORD_KEY], axis=1, inplace=True)
        self.x_test.drop([keys.PLAN_CODE_KEY, keys.RECORD_KEY, keys.PLAN_CODE_RECORD_KEY], axis=1, inplace=True)
        norm = MinMaxScaler().fit(self.x_train)
        self.x_train_norm = norm.transform(self.x_train)
        self.x_test_norm = norm.transform(self.x_test)

        self.models_developed.append(GradientBoostingClassifier(random_state=123).fit(self.x_train, self.y_train))
        best_hyperparameters_RF = self.get_best_hyperparameters_RandomForest()
        self.models_developed.append(RandomForestClassifier(
            max_depth=best_hyperparameters_RF.max_depth,
            n_estimators=best_hyperparameters_RF.n_estimators,
            random_state=123).fit(self.x_train, self.y_train))
        best_hyperparameters_SVM = self.get_best_hyperparameters_SVM()
        self.models_developed.append(SVC(
            C=best_hyperparameters_SVM.C,
            gamma=best_hyperparameters_SVM.gamma,
            probability=True, random_state=123).fit(self.x_train_norm, self.y_train))

        pred_1 = self.models_developed[0].predict_proba(self.x_test)
        pred_2 = self.models_developed[1].predict_proba(self.x_test)
        pred_3 = self.models_developed[2].predict_proba(self.x_test_norm)
        weighted_ensemble_pred = pred_1 * 0.7 + pred_2 * 0.15 + pred_3 * 0.15
        y_pred = (weighted_ensemble_pred[0:, 1] >= self.threshold).astype(int)

        log.info("accuracy of model is: " + str(sklm.accuracy_score(y_true=self.y_test, y_pred=y_pred)))
        log.info("confusion matrix of model is: \n" + str(sklm.confusion_matrix(y_true=self.y_test, y_pred=y_pred)))
        log.info("recall of model is: " + str(sklm.recall_score(y_true=self.y_test, y_pred=y_pred)))

        self.input_dfs[0].drop([keys.PLAN_CODE_KEY, keys.RECORD_KEY, keys.PLAN_CODE_RECORD_KEY], axis=1, inplace=True)
        x_test = self.input_dfs[0].drop([keys.DROP_OUT_KEY], axis=1)
        y_test = self.input_dfs[0][keys.DROP_OUT_KEY]
        x_test_norm = norm.transform(x_test)

        pred_1 = self.models_developed[0].predict_proba(x_test)
        pred_2 = self.models_developed[1].predict_proba(x_test)
        pred_3 = self.models_developed[2].predict_proba(x_test_norm)
        weighted_ensemble_pred = pred_1 * 0.7 + pred_2 * 0.15 + pred_3 * 0.15
        y_pred = (weighted_ensemble_pred[0:, 1] >= self.threshold).astype(int)

        log.info("accuracy of model with complete data is: " + str(sklm.accuracy_score(y_true=y_test, y_pred=y_pred)))
        log.info("confusion matrix of model with complete data is: \n" + str(sklm.confusion_matrix(y_true=y_test,
                                                                                                   y_pred=y_pred)))
        log.info("recall of model with complete data is: " + str(sklm.recall_score(y_true=y_test, y_pred=y_pred)))

        self.y_pred = y_pred

    def save(self):

        output_path = Path(self.output_path_segment)
        if not output_path.exists():
            output_path.mkdir(parents=True)

        path_segment = str(output_path)
        output_path_model = Path(path_segment) / 'models'
        if not output_path_model.exists():
            output_path_model.mkdir(parents=True)
        output_path_model = Path(output_path_model) / str(self.school_kind.value + '_gradientBoosting_model')
        output_path_model = output_path_model.with_suffix(".sav")
        pickle.dump(self.models_developed[0], open(output_path_model, 'wb'))

        path_segment = str(output_path)
        output_path_model = Path(path_segment) / 'models'
        output_path_model = Path(output_path_model) / str(self.school_kind.value + '_randomForest_model')
        output_path_model = output_path_model.with_suffix(".sav")
        pickle.dump(self.models_developed[1], open(output_path_model, 'wb'))

        path_segment = str(output_path)
        output_path_model = Path(path_segment) / 'models'
        output_path_model = Path(output_path_model) / str(self.school_kind.value + '_SVM_model')
        output_path_model = output_path_model.with_suffix(".sav")
        pickle.dump(self.models_developed[2], open(output_path_model, 'wb'))

        path_segment = str(output_path)
        output_path_model = Path(path_segment) / 'train_ids'
        if not output_path_model.exists():
            output_path_model.mkdir(parents=True)
        output_path_model = Path(output_path_model) / str(self.school_kind.value + '_train_ids')
        output_path_model = output_path_model.with_suffix(".csv")
        self.train_ids.to_csv(
            output_path_model,
            sep=self.output_separator,
            index=False)


def main():
    logging.basicConfig(
        filename="ensemble_model_before_predict_debug.log",
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
