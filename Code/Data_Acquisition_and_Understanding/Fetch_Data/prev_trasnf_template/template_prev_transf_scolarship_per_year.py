import pandas as pd
import apitep_core


class TemplatePreviousTransScholarshipPerYear:

    @staticmethod
    def execute_previous_transf(input_path: str, output_path: str, number_of_sheets: int):
        polytechnic_scholarship_per_year = ""
        i = 1
        while i <= number_of_sheets:
            polytechnic_scholarship_per_year_n = pd.read_excel(
                input_path,
                sheet_name='Sheet ' + str(i),
                header=0)
            if i == 1:
                polytechnic_scholarship_per_year = polytechnic_scholarship_per_year_n.copy()
            else:
                polytechnic_scholarship_per_year = pd.concat([polytechnic_scholarship_per_year,
                                                              polytechnic_scholarship_per_year_n])
            i += 1

        polytechnic_scholarship_per_year = polytechnic_scholarship_per_year[['expediente', 'cod_plan', 'des_plan',
                                                                             'curso_academico', 'becario']]
        polytechnic_scholarship_per_year.to_csv(output_path,
                                                index=False,
                                                header=True,
                                                sep=apitep_core.CSV_SEPARATOR)
