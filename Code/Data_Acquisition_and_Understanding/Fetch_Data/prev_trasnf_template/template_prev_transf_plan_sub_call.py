import pandas as pd
import apitep_core


class TemplatePreviousTransfPlanSubCall:

    @staticmethod
    def execute_previous_transf(input_path: str, output_path: str, number_of_sheets: int):
        plan_subject_call = ""
        i = 1
        while i <= number_of_sheets:
            plan_subject_call_n = pd.read_excel(
                input_path,
                sheet_name='Sheet ' + str(i),
                header=0)
            if i == 1:
                plan_subject_call = plan_subject_call_n.copy()
            else:
                plan_subject_call = pd.concat([plan_subject_call, plan_subject_call_n])
            i += 1

        plan_subject_call = plan_subject_call[['expediente', 'cod_plan', 'des_plan', 'cod_asignatura',
                                               'des_asignatura', 'curso_asignatura', 'semestre', 'tipo_linea',
                                               'tipologia_asignatura', 'des_tipogia_asignatura', 'curso_academico',
                                               'convocatoria', 'nota_alf', 'nota_num', 'des_nota',
                                               'numero_convocatorias_agotadas', 'fin_estudios']]
        plan_subject_call.to_csv(output_path,
                                 index=False,
                                 header=True,
                                 sep=apitep_core.CSV_SEPARATOR)
