from template_prev_transf_plan_sub_call import TemplatePreviousTransfPlanSubCall


def main():
    # TODO: CAMBIAR PATH POR EL DEL SERVIDOR CUANDO ALOJEMOS LOS DATOS EN EL SERVIDOR.
    TemplatePreviousTransfPlanSubCall.execute_previous_transf(
        input_path='/media/fran/Datos/Vicerrectorado/Datos UEx/politecnica/Politecnica -plan asignatura '
                   'convocatoria.xls',
        header_1=1,
        header_2=0,
        header_3=0,
        output_path='../../../Data/Interim/plan_subject_call.csv')


if __name__ == "__main__":
    main()
