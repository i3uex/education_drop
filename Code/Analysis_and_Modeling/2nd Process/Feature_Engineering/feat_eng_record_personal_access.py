from apitep_utils.feature_engineering import FeatureEngineering
import logging
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

        self.input_df['tipo_traslado'] = self.input_df['tipo_traslado'].apply(
            lambda func: 'N' if pd.isna(func) else func)

        analys_columns = ['des_plan', 'anio_apertura_expediente', 'abandona', 'convocatoria_acceso', 'des_acceso',
                          'nota_admision_def', 'sexo', 'anio_nacimiento', 'provincia', 'municipio']

        analys_record_personal_access = self.input_df[analys_columns]
        analys_record_personal_access.dropna(inplace=True)

        self.output_df = analys_record_personal_access


def main():
    logging.basicConfig(
        filename="debug.log",
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
