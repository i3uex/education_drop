import pickle
from typing import List

import argparse

import numpy as np
import pandas as pd
from apitep_utils import ArgumentParserHelper
from apitep_utils.analysis_modelling import AnalysisModeling
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as sklm

import keys
from data_model.school_kind import SchoolKind
import logging

log = logging.getLogger(__name__)


class PredictDV(AnalysisModeling):
    quadrimester: int = None
    course: int = None
    school_kind: SchoolKind = None
    threshold: float = 0.47
    input_path_models: List = None
    final_analys_record_personal_access: pd.DataFrame = None
    y_pred: pd.Series = None
    input_dfs: List = []
    input_path_segments: List = None
    model_number: str = None
    df: pd.DataFrame = None

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
        argument_parser.add_argument("-im", "--input_models", required=True,
                                     nargs="+",
                                     help="paths to the inputs models")
        argument_parser.add_argument("-mn", "--model_number", required=True,
                                     help="number of model to create")
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

        input_path_models = arguments.input_models
        self.input_path_models = []
        for input_path_model in input_path_models:
            self.input_path_models.append(
                ArgumentParserHelper.parse_data_file_path(
                    data_file_path=input_path_model)
            )
        self.model_number = arguments.model_number

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
                                '2nd_model','cum_absent_ratio', 'cum_pass_ratio', 'nota_admision_def']
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
                                'cum_median', '4th_model', '5th_model', 'nota_admision_def', 'cum_more_1st_call_ratio',
                                'cum_absent_ratio', 'cum_pass_ratio']
            elif self.course == 3 and self.quadrimester == 2:
                final_colums = ['des_plan', 'anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'anio_nacimiento_interval', '1st_model', 'scholarship', '2nd_model', '3rd_model',
                                'cum_median', '4th_model', '5th_model', '6th_model', 'nota_admision_def',
                                'cum_more_1st_call_ratio', 'cum_absent_ratio', 'cum_pass_ratio']
            elif self.course == 4 and self.quadrimester == 1:
                final_colums = ['anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'anio_nacimiento_interval', '1st_model', 'scholarship', '2nd_model', '3rd_model',
                                'cum_median', '4th_model', '5th_model', '6th_model', '7th_model', 'cum_absent_ratio',
                                'cum_pass_ratio', 'nota_admision_def', 'cum_more_1st_call_ratio']
            elif self.course == 4 and self.quadrimester == 2:
                final_colums = ['anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                                'anio_nacimiento_interval', '1st_model', 'scholarship', '2nd_model', '3rd_model',
                                'cum_median', '4th_model', '5th_model', '6th_model', '7th_model', '8th_model',
                                'cum_absent_ratio', 'cum_pass_ratio', 'cum_more_1st_call_ratio',
                                'nota_admision_def']
            else:
                raise NotImplementedError

        final_colums.append(keys.RECORD_KEY)
        final_colums.append(keys.PLAN_CODE_KEY)
        self.input_dfs[0].sort_values(by=[keys.PLAN_CODE_KEY, keys.RECORD_KEY], inplace=True)
        self.final_analys_record_personal_access = self.input_dfs[0].copy()
        self.input_dfs[0] = self.input_dfs[0][final_colums]

        if keys.CUM_MEDIAN_KEY in self.input_dfs[0].columns:
            median_bcket_array = np.array([0, 1.5, 3, 4.5, 6, 7.5, 9, 10])
            self.input_dfs[0][keys.CUM_MEDIAN_INTERVAL_KEY] = pd.cut(
                self.input_dfs[0][keys.CUM_MEDIAN_KEY], median_bcket_array, include_lowest=True)
            self.input_dfs[0].drop([keys.CUM_MEDIAN_KEY], axis=1, inplace=True)

        # TRATAMIENTO DEL DATASET DE ENTRENAMIENTO

        if keys.CUM_MEDIAN_KEY in self.input_dfs[1].columns:
            median_bcket_array = np.array([0, 1.5, 3, 4.5, 6, 7.5, 9, 10])
            self.input_dfs[1][keys.CUM_MEDIAN_INTERVAL_KEY] = pd.cut(
                self.input_dfs[1][keys.CUM_MEDIAN_KEY], median_bcket_array, include_lowest=True)
            self.input_dfs[1].drop([keys.CUM_MEDIAN_KEY], axis=1, inplace=True)

        # CONCATENACIÓN CON DATASET DE ENTRENAMIENTO
        self.input_dfs[0][keys.PLAN_CODE_RECORD_KEY] = self.input_dfs[0].apply(
            lambda func: int(str(func[keys.PLAN_CODE_KEY]) + str(func[keys.RECORD_KEY])), axis=1)
        self.input_dfs[1][keys.PLAN_CODE_RECORD_KEY] = self.input_dfs[1].apply(
            lambda func: int(str(func[keys.PLAN_CODE_KEY]) + str(func[keys.RECORD_KEY])), axis=1)
        valid_rows = self.input_dfs[0][keys.PLAN_CODE_RECORD_KEY]
        no_valid_rows = self.input_dfs[1][keys.PLAN_CODE_RECORD_KEY]

        self.input_dfs[0] = pd.concat([self.input_dfs[0], self.input_dfs[1]], ignore_index=False)
        self.input_dfs[0].drop_duplicates(subset=[keys.PLAN_CODE_RECORD_KEY], keep='last', inplace=True)
        self.input_dfs[0] = pd.get_dummies(data=self.input_dfs[0],
                                           columns=self.input_dfs[0].drop(self.input_dfs[0].select_dtypes(
                                               include=['int64', 'float64']).columns, axis=1).columns)
        self.df = self.input_dfs[0][self.input_dfs[0][keys.PLAN_CODE_RECORD_KEY].isin(no_valid_rows)]
        self.input_dfs[0] = self.input_dfs[0][self.input_dfs[0][keys.PLAN_CODE_RECORD_KEY].isin(valid_rows)]
        self.input_dfs[0].sort_values(by=[keys.PLAN_CODE_KEY, keys.RECORD_KEY], inplace=True)
        self.df.sort_values(by=[keys.PLAN_CODE_KEY, keys.RECORD_KEY], inplace=True)

        self.input_dfs[0].drop([keys.PLAN_CODE_KEY, keys.RECORD_KEY, keys.PLAN_CODE_RECORD_KEY], axis=1, inplace=True)
        self.df.drop([keys.PLAN_CODE_KEY, keys.RECORD_KEY, keys.PLAN_CODE_RECORD_KEY], axis=1, inplace=True)

    def analise(self):
        self.models_developed.append(pickle.load(open(self.input_path_models[0], 'rb')))
        self.models_developed.append(pickle.load(open(self.input_path_models[1], 'rb')))
        self.models_developed.append(pickle.load(open(self.input_path_models[2], 'rb')))

        x_test = self.input_dfs[0].drop([keys.DROP_OUT_KEY], axis=1)
        norm = MinMaxScaler().fit(x_test)
        x_test_norm = norm.transform(x_test)

        pred_1 = self.models_developed[0].predict_proba(x_test)
        pred_2 = self.models_developed[1].predict_proba(x_test)
        pred_3 = self.models_developed[2].predict_proba(x_test_norm)
        weighted_ensemble_pred = pred_1 * 0.7 + pred_2 * 0.15 + pred_3 * 0.15
        self.y_pred = (weighted_ensemble_pred[0:, 1] >= self.threshold).astype(int)

        x_test = self.df.drop([keys.DROP_OUT_KEY], axis=1)
        norm = MinMaxScaler().fit(x_test)
        x_test_norm = norm.transform(x_test)

        pred_1 = self.models_developed[0].predict_proba(x_test)
        pred_2 = self.models_developed[1].predict_proba(x_test)
        pred_3 = self.models_developed[2].predict_proba(x_test_norm)
        weighted_ensemble_pred = pred_1 * 0.7 + pred_2 * 0.15 + pred_3 * 0.15
        y_pred = (weighted_ensemble_pred[0:, 1] >= self.threshold).astype(int)
        y_test = self.df[keys.DROP_OUT_KEY]

        log.info("accuracy of model with complete data is: " + str(sklm.accuracy_score(y_true=y_test, y_pred=y_pred)))
        log.info("confusion matrix of model with complete data is: \n" + str(sklm.confusion_matrix(y_true=y_test,
                                                                                                   y_pred=y_pred)))
        log.info("recall of model with complete data is: " + str(sklm.recall_score(y_true=y_test, y_pred=y_pred)))

    def save(self):
        self.final_analys_record_personal_access[self.model_number] = self.y_pred
        self.final_analys_record_personal_access[keys.PLAN_CODE_RECORD_KEY] = self.final_analys_record_personal_access. \
            apply(lambda func: int(str(func[keys.PLAN_CODE_KEY]) + str(func[keys.RECORD_KEY])), axis=1)
        self.final_analys_record_personal_access[keys.TRAIN_KEY] = self.final_analys_record_personal_access[
            keys.PLAN_CODE_RECORD_KEY].apply(
            lambda func: True if func in list(self.input_dfs[2]['cod_plan_exp']) else False)

        self.final_analys_record_personal_access.to_csv(
            self.output_path_segment,
            sep=self.output_separator,
            index=False)


def main():
    logging.basicConfig(
        filename="dv_prediction_debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("--------------------------------------------------------------------------------------")
    log.info("Start PredictDV")
    log.debug("main()")

    analys = PredictDV(
        input_separator='|',
        output_separator='|'
    )
    analys.execute()


if __name__ == "__main__":
    main()
