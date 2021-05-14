from template_prev_transf_scolarship_per_year import TemplatePreviousTransScholarshipPerYear


def main():
    # TODO: CAMBIAR PATH POR EL DEL SERVIDOR CUANDO ALOJEMOS LOS DATOS EN EL SERVIDOR.
    TemplatePreviousTransScholarshipPerYear.execute_previous_transf(
        input_path='/media/fran/Datos/Vicerrectorado/Datos UEx/politecnica/Politecnica -becario por curso academico.xls',
        output_path='../../../Data/Interim/Polytechnic/polytechnic_scholarship_per_year.csv',
        number_of_sheets=1
    )
    TemplatePreviousTransScholarshipPerYear.execute_previous_transf(
        input_path='/media/fran/Datos/Vicerrectorado/Datos UEx/profesorado/Profesorado -becario por curso academico.xls',
        output_path='../../../Data/Interim/Teaching/teaching_scholarship_per_year.csv',
        number_of_sheets=1
    )


if __name__ == "__main__":
    main()
