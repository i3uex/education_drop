import pandas as pd
import apitep_core
from template_prev_transf_record_personal_access import TemplatePreviousTransfRecordPersonalAccess


def main():
    # TODO: CAMBIAR PATH POR EL DEL SERVIDOR CUANDO ALOJEMOS LOS DATOS EN EL SERVIDOR.
    TemplatePreviousTransfRecordPersonalAccess.execute_previous_transf(
        input_path='/media/fran/Datos/Vicerrectorado/Datos UEx/politecnica/Politecnica -datos de expediente,'
                   'personales y de acceso.xls',
        output_path='../../../Data/Interim/Polytechnic/polytechnic_record_personal_access.csv',
        number_of_sheets=1
    )
    TemplatePreviousTransfRecordPersonalAccess.execute_previous_transf(
        input_path='/media/fran/Datos/Vicerrectorado/Datos UEx/profesorado/Profesorado -datos de expediente,'
                   'personales y de acceso.xls',
        output_path='../../../Data/Interim/Teaching/teaching_record_personal_access.csv',
        number_of_sheets=1
    )


if __name__ == "__main__":
    main()
