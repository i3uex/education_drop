import pandas as pd
import apitep_core


class TemplatePreviousTransfPlanSubCall:

    @staticmethod
    def execute_previous_transf(input_path: str, header_1: int, header_2: int, header_3: int, output_path: str):
        plan_subject_call_1 = pd.read_excel(
            input_path,
            sheet_name='Sheet 1',
            header=header_1)
        plan_subject_call_2 = pd.read_excel(
            input_path,
            sheet_name='Sheet 2',
            header=header_2)
        plan_subject_call_3 = pd.read_excel(
            input_path,
            sheet_name='Sheet 3',
            header=header_3)
        plan_subject_call = pd.concat([plan_subject_call_1, plan_subject_call_2, plan_subject_call_3])
        plan_subject_call = plan_subject_call[['expediente', 'cod_plan', 'des_plan', 'cod_asignatura',
                                               'des_asignatura', 'curso_asignatura', 'semestre', 'tipo_linea',
                                               'tipologia_asignatura', 'des_tipogia_asignatura', 'curso_academico',
                                               'convocatoria', 'nota_alf', 'nota_num', 'des_nota',
                                               'numero_convocatorias_agotadas', 'fin_estudios']]
        plan_subject_call.to_csv(output_path,
                                 index=False,
                                 header=True,
                                 sep=apitep_core.CSV_SEPARATOR)
