import pandas as pd
import apitep_core
from template_prev_transf_record_personal_access import TemplatePreviousTransfRecordPersonalAccess


def main():
    # TODO: CAMBIAR PATH POR EL DEL SERVIDOR CUANDO ALOJEMOS LOS DATOS EN EL SERVIDOR.
    TemplatePreviousTransfRecordPersonalAccess.execute_previous_transf(
        input_path='/media/fran/Datos/Vicerrectorado/Datos UEx/politecnica/Politecnica -datos de expediente,'
                   'personales y de acceso.xls',
        header=1,
        output_path='../../../Data/Interim/polytechnic_record_personal_access.csv'
    )


if __name__ == "__main__":
    main()
