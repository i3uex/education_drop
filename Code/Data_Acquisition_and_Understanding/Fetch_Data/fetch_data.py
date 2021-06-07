import logging
from apitep_utils import ETL

log = logging.getLogger(__name__)


class FetchData(ETL):

    @ETL.stopwatch
    def process(self):
        """
        Process raw data of education drop
        """
        columns = list(self.input_df.columns)
        columns.remove("expediente")
        columns.insert(0, "expediente")

        log.info("reorder columns of dataset and set on top column 'expediente'")

        self.output_df = self.input_df[columns]


def main():
    logging.basicConfig(
        filename="debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("Start FetchData")
    log.debug("main()")

    etl = FetchData(
        save_report_on_load=False,
        save_report_on_save=False,
        input_type_excel=True,
        output_separator='|'
    )

    etl.execute()


if __name__ == "__main__":
    main()
