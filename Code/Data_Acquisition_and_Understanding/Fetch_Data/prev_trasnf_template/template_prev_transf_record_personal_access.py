import pandas as pd
import apitep_core


class TemplatePreviousTransfRecordPersonalAccess:

    @staticmethod
    def execute_previous_transf(input_path: str, header: int, output_path: str):
        polytechnic_record_personal_access = pd.read_excel(
            input_path,
            sheet_name='Sheet 1',
            header=header)
        polytechnic_record_personal_access = polytechnic_record_personal_access[
            ['expediente', 'cod_plan', 'des_plan', 'anio_apertura_expediente',
             'exp_cerrado', 'exp_trasladado', 'tipo_traslado', 'exp_bloqueado',
             'anio_convocatoria_acceso', 'convocatoria_acceso', 'acceso',
             'des_acceso', 'sub_acceso', 'des_subacesso', 'nota_acceso',
             'nota_admision_def', 'centro_escolar_acceso', 'sexo',
             'fecha_nacimiento', 'cod_provincia', 'provincia', 'cod_municipio',
             'municipio']]
        polytechnic_record_personal_access.to_csv(output_path,
                                                  index=False,
                                                  header=True,
                                                  sep=apitep_core.CSV_SEPARATOR)
