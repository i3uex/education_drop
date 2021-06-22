from apitep_utils.integration import Integration
import logging
import pandas as pd

log = logging.getLogger(__name__)


class RecordPersonalAccessIntegration(Integration):

    @Integration.stopwatch
    def process(self):
        """
        Integration record_personal_access data
        """

        log.info("Integration UniversityDegrees data")
        log.debug("UniversityDegreesIntegration.process()")

        pr_polytechnic_record_personal_access = self.input_dfs[0]
        pr_teaching_record_personal_access = self.input_dfs[1]

        pr_polytechnic_record_personal_access['facultad'] = 'POLITECNICA'
        pr_teaching_record_personal_access['facultad'] = 'PROFESORADO'

        self.output_df = pd.concat([pr_polytechnic_record_personal_access, pr_teaching_record_personal_access], axis=0)
        log.info("columns of final dataset are:" + self.output_df.columns)


def main():
    """
    Set logging up. Integration data
    :return:
    """

    logging.basicConfig(
        filename="debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("Start RecordPersonalAccessIntegration")
    log.debug("main()")

    integration = RecordPersonalAccessIntegration(
        input_separator='|',
        output_separator='|',
        report_type=Integration.ReportType.Both
    )
    integration.execute()


if __name__ == "__main__":
    main()
