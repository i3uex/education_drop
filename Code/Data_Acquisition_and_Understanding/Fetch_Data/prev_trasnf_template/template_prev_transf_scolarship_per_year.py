import pandas as pd
import apitep_core


class TemplatePreviousTransScholarshipPerYear:

    @staticmethod
    def execute_previous_transf(input_path: str, header: int, output_path: str):
        polytechnic_scholarship_per_year = pd.read_excel(
            input_path,
            sheet_name='Sheet 1',
            header=header)
        polytechnic_scholarship_per_year = polytechnic_scholarship_per_year[['expediente', 'cod_plan', 'des_plan',
                                                                             'curso_academico', 'becario']]
        polytechnic_scholarship_per_year.to_csv(output_path,
                                                index=False,
                                                header=True,
                                                sep=apitep_core.CSV_SEPARATOR)
