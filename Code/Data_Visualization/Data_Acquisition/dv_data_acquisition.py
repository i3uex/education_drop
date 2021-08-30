import argparse
from apitep_utils import ETL, ArgumentParserHelper
import logging

import keys
from data_model.school_kind import SchoolKind

log = logging.getLogger(__name__)


class DataAcquisitionDV(ETL):
    school_kind: SchoolKind = None

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

        program_description = self.description
        argument_parser = argparse.ArgumentParser(description=program_description)
        argument_parser.add_argument("-i", "--input_paths",
                                     required=True,
                                     nargs="+",
                                     help="path to the input CSV datasets")
        argument_parser.add_argument("-o", "--output_path", required=True,
                                     help="path to the output CSV dataset")
        argument_parser.add_argument("-s", "--school_kind", required=True,
                                     help="school kind to analyze")

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
        school_kind_str = arguments.school_kind
        if school_kind_str == "Teaching":
            self.school_kind = SchoolKind.Teaching
        elif school_kind_str == "Polytechnic":
            self.school_kind = SchoolKind.Polytechnic

    @ETL.stopwatch
    def process(self):
        """
        Process record personal data
        """
        log.info("Data acquisition of record_personal_access data for data visualization of school: " +
                 self.school_kind.value)
        log.debug("DataAcquisitionDV.process()")

        target_courses = ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21']

        rows_before = len(self.input_dfs[0].index)
        self.input_dfs[0] = self.input_dfs[0][self.input_dfs[0][keys.OPEN_YEAR_PLAN_KEY].isin(target_courses)]
        rows_after = len(self.input_dfs[0].index)
        self.changes["get only target courses for visualization"] = rows_before - rows_after

        log.info("columns of final dataset are:" + str(self.input_dfs[0].columns))
        log.info("final number of rows: " + str(len(self.input_dfs[0].index)))
        self.output_df = self.input_dfs[0]


def main():
    logging.basicConfig(
        filename="debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("--------------------------------------------------------------------------------------")
    log.info("Start DataAcquisitionDV")
    log.debug("main()")

    etl = DataAcquisitionDV(
        input_separator="|",
        output_separator="|",
        save_report_on_save=False,
        save_report_on_load=False,
    )
    etl.execute()


if __name__ == "__main__":
    main()
