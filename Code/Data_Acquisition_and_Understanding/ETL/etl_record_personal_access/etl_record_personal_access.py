import logging
from apitep_utils import ETL
import pandas as pd

log = logging.getLogger(__name__)


class RecordPersonalAccessETL(ETL):

    @ETL.stopwatch
    def process(self):
        """
        Process record_personal_access data
        """

        log.info("Process record_personal_access data")
        log.debug("RecordPersonalAccessETL.process()")
        self.output_df=self.input_df


def main():
    logging.basicConfig(
        filename="debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("Start RecordPersonalAccessETL")
    log.debug("main()")

    etl = RecordPersonalAccessETL(
        input_separator="|",
        output_separator="|",
        report_type=ETL.ReportType.Both,
        save_report_on_load=False
    )
    etl.parse_arguments()
    etl.load()
    etl.process()
    etl.save()
    etl.log_changes()


if __name__ == "__main__":
    main()
