from apitep_utils.feature_engineering import FeatureEngineering
import logging

import keys
import numpy as np
import pandas as pd
log = logging.getLogger(__name__)


class RecordPersonalAccessFeatureEngineering(FeatureEngineering):

    @FeatureEngineering.stopwatch
    def process(self):
        """
        Feature Engineering int_record_personal_access
        """

        log.info("Feature Engineering of int_record_personal_access data")
        log.debug("RecordPersonalAccessFeatureEngineering.process()")

        analys_columns = [keys.RECORD_KEY, keys.PLAN_CODE_KEY, keys.PLAN_DESCRIPTION_KEY, keys.OPEN_YEAR_PLAN_KEY,
                          keys.DROP_OUT_KEY, keys.ACCESS_CALL_KEY, keys.ACCESS_DESCRIPTION_KEY,
                          keys.FINAL_ADMISION_NOTE_KEY, keys.GENDER_KEY, keys.BIRTH_YEAR_KEY,
                          keys.PROVINCE_KEY, keys.TOWN_KEY]

        self.input_dfs[0] = self.input_dfs[0][analys_columns]
        self.input_dfs[0].dropna(inplace=True)

        anio_nacimiento_bcket_array = np.linspace(1960, 2000, 9)
        self.input_dfs[0][keys.BIRTH_YEAR_INTERVAL_KEY] = pd.cut(
            self.input_dfs[0][keys.BIRTH_YEAR_KEY], anio_nacimiento_bcket_array, include_lowest=True)
        self.input_dfs[0].drop([keys.BIRTH_YEAR_KEY], axis=1, inplace=True)

        self.input_dfs[0]['abandona'] = self.input_dfs[0]['abandona'].apply(
            lambda func: 1 if func == 'S' else 0)

        self.output_df = self.input_dfs[0]


def main():
    logging.basicConfig(
        filename="personal_access_debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("Start RecordPersonalAccessFeatureEngineering")
    log.debug("main()")

    feature_eng = RecordPersonalAccessFeatureEngineering(
        input_separator='|',
        output_separator='|'
    )
    feature_eng.execute()


if __name__ == "__main__":
    main()
