from template_prev_transf_plan_sub_call import TemplatePreviousTransfPlanSubCall


def main():
    # TODO: CAMBIAR PATH POR EL DEL SERVIDOR CUANDO ALOJEMOS LOS DATOS EN EL SERVIDOR.
    TemplatePreviousTransfPlanSubCall.execute_previous_transf(
        input_path='/media/fran/Datos/Vicerrectorado/Datos UEx/politecnica/Politecnica -plan asignatura '
                   'convocatoria.xls',
        output_path='../../../Data/Interim/Polytechnic/polytechnic_plan_subject_call.csv',
        number_of_sheets=3)

    TemplatePreviousTransfPlanSubCall.execute_previous_transf(
        input_path='/media/fran/Datos/Vicerrectorado/Datos UEx/profesorado/Profesorado -plan asignatura '
                   'convocatoria.xls',
        output_path='../../../Data/Interim/Teaching/teaching_plan_subject_call.csv',
        number_of_sheets=4)


if __name__ == "__main__":
    main()
