import logging
from apitep_utils import ETL
import pandas as pd
from apitep_utils.transformation import Transformation

import keys
from data_model.titulation_kind import SchoolKind

log = logging.getLogger(__name__)


class ScolarshipPerYearETL(ETL):
    school_kind: SchoolKind = None

    def __init__(
            self,
            input_path_segment: str = None,
            output_path_segment: str = None,
            input_separator: str = None,
            output_separator: str = None,
            save_report_on_load: bool = None,
            save_report_on_save: bool = None,
            report_type: Transformation.ReportType = None,
            report_path_segment: str = None,
            input_type_excel: bool = None,
            school_kind: SchoolKind = None,
    ):
        super().__init__(
            input_path_segment=input_path_segment,
            output_path_segment=output_path_segment,
            input_separator=input_separator,
            output_separator=output_separator,
            save_report_on_load=save_report_on_load,
            save_report_on_save=save_report_on_save,
            report_type=report_type,
            report_path_segment=report_path_segment,
            input_type_excel=input_type_excel
        )

        if school_kind is not None:
            self.school_kind = school_kind

    @ETL.stopwatch
    def process(self):
        """
        Process scholarship_per_year data
        """

        log.info("Process scholarship_per_year data")
        log.debug("ScolarshipPerYearETL.process()")

        if self.school_kind is SchoolKind.Polytechnic:
            old_degrees = ['MÁSTER UNIVERSITARIO EN COMPUTACIÓN GRID Y PARALELISMO',
                           'MÁSTER DE ESPECIALIZACIÓN EN GEOTECNOLOGÍAS TOPOGRÁFICAS EN LA INGENIERÍA',
                           'MÁSTER UNIVERSITARIO EN EVALUACIÓN Y GESTIÓN DEL RUIDO AMBIENTAL',
                           'MÁSTER EN COMPUTACIÓN GRID Y PARALELISMO']
        elif self.school_kind is SchoolKind.Teaching:
            old_degrees = ['MÁSTER UNIVERSITARIO EN INVESTIGACIÓN EN CIENCIAS SOCIALES Y JURÍDICAS',
                           'MÁSTER UNIV. FORMACIÓN EN PORTUGUÉS PARA PROF. ENSEÑANZA PRIM. Y SECUNDARIA']
        else:
            raise NotImplementedError

        rows_before = len(self.input_df.index)
        self.input_df = self.input_df[self.input_df[keys.PLAN_DESCRIPTION_KEY].isin(old_degrees) == False]
        rows_after = len(self.input_df.index)
        self.changes["delete data of old degrees"] = rows_before - rows_after

        log.info("columns of final dataset are:" + self.input_df.columns)

        self.output_df = self.input_df


def main():
    school_kind = SchoolKind.Polytechnic

    if school_kind is SchoolKind.Polytechnic:
        logging.basicConfig(
            filename="pol_debug.log",
            level=logging.DEBUG,
            format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
    elif school_kind is school_kind.Teaching:
        logging.basicConfig(
            filename="teach_debug.log",
            level=logging.DEBUG,
            format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
    else:
        raise NotImplementedError

    etl = ScolarshipPerYearETL(
        input_separator="|",
        output_separator="|",
        report_type=ETL.ReportType.Both,
        school_kind=school_kind
    )
    etl.execute()


if __name__ == "__main__":
    main()
