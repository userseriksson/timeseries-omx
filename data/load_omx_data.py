import pandas as pd
import yfinance as yf
import datetime
import time
import requests
import io

start = datetime.datetime(2020,4,1)
end = datetime.datetime(2021,4,1)

Symbols = ['SEB-A.ST','ABB.ST','SAND.ST','ICA.ST','TELIA.ST']

'''Symbols = [[['8TRA.ST',	'AAK.ST',	'ABB.ST',	'ADDTB.ST',	'AFRY.ST',	'ALFA.ST',	'ARJOB.ST',	'ASSAB.ST',	'ATCOA.ST',	'ATCOB.ST',
       'ATRLJB.ST',	'AXFO.ST',	'AZA.ST',	'AZN.ST',	'BALDB.ST',	'BEIJB.ST',	'BETSB.ST',	'BHG.ST',	'BILL.ST',	'BOL.ST',	'BRAV.ST',
       'BURE.ST',	'CAST.ST',	'CATE.ST',	'CINT.ST',	'DOM.ST',	'EKTAB.ST',	'ELUXA.ST',	'ELUXB.ST',	'EPIA.ST',	'EPIB.ST',	'EPROB.ST',	
       'EQT.ST',	'ERICA.ST',	'ERICB.ST',	'ESSITYA.ST',	'ESSITYB.ST',	'EVO.ST',	'FABG.ST',	'FOIB.ST',	'FPARA.ST',	'FPARD.ST',	'FPARPREF.ST',
       'GETIB.ST',	'HEM.ST',	'HEXAB.ST',	'HMB.ST',	'HOLMA.ST',	'HOLMB.ST',	'HPOLB.ST',	'HUFVA.ST',	'HUSQA.ST',	'HUSQB.ST',	'ICA.ST',	'INDT.ST',
       'INDUA.ST',	'INDUC.ST',	'INTRUM.ST',	'INVEA.ST',	'INVEB.ST',	'JM.ST',	'KINDSDB.ST',	'KINVA.ST',	'KINVB.ST',	'KLED.ST',	'KLOVA.ST',	
       'KLOVB.ST',	'KLOVPREF.ST',	'LATOB.ST',	'LIFCOB.ST',	'LOOMIS.ST',	'LUMI.ST',	'LUNDB.ST',	'LUNE.ST',	'MCOVB.ST',	'MYCR.ST',	'NCCA.ST',
       'NCCB.ST',	'NDASE.ST',	'NENTA.ST',	'NENTB.ST',	'NIBEB.ST',	'NOBI.ST',	'NOLAB.ST',	'NYF.ST',	'PEABB.ST',	'PLAZB.ST',	'PNDXB.ST',	'RATOA.ST',	
       'RATOB.ST',	'RESURS.ST',	'SAABB.ST',	'SAGAA.ST',	'SAGAB.ST',	'SAGAD.ST',	'SAND.ST',	'SAVE.ST',	'SBBB.ST',	'SBBD.ST',	'SCAA.ST',
       'SCAB.ST',	'SDIPB.ST',	'SDIPPREF.ST',	'SEBA.ST',	'SEBC.ST',	'SECTB.ST',	'SECUB.ST',	'SF.ST',	'SHBA.ST',	'SHBB.ST',	'SINCH.ST',
       'SKAB.ST',	'SKFA.ST',	'SKFB.ST',	'SOBI.ST',	'SSABA.ST',	'SSABB.ST',	'STEA.ST',	'STER.ST',	'SWECA.ST',	'SWECB.ST',	'SWEDA.ST',	'SWMA.ST',
       'TEL2A.ST',	'TEL2B.ST',	'TELIA.ST',	'THULE.ST',	'TIETOS.ST',	'TIGOSDB.ST',	'TRELB.ST',	'VITR.ST',	'VNESDB.ST',	'VOLVA.ST',	'VOLVB.ST',	
       'WALLB.ST',	'WIHL.ST',	'ACAD.ST',	'ADAPT.ST',	'ALIFB.ST',	'ALIG.ST',	'AMBEA.ST',	'ANNEB.ST',	'ANODB.ST',	'AOI.ST',	'AQ.ST',	'ATT.ST',	
       'BACTIB.ST',	'BALCO.ST',	'BEIAB.ST',	'BERGB.ST',	'BESQ.ST',	'BETCO.ST',	'BILIA.ST',	'BIOAB.ST',	'BIOGB.ST',	'BIOT.ST',	'BMAX.ST',	'BONAVA.ST',
       'BONAVB.ST',	'BONEX.ST',	'BOOZT.ST',	'BRINB.ST',	'BTSB.ST',	'BUFAB.ST',	'BULTEN.ST',	'CALTX.ST',	'CAMX.ST',	'CANTA.ST',	'CATA.ST',	'CATB.ST',
       'CCC.ST',	'CEVI.ST',	'CIBUS.ST',	'CLAB.ST',	'CLASB.ST',	'CLNKB.ST',	'COIC.ST',	'COLL.ST',	'COOR.ST',	'COREA.ST',	'COREB.ST',	'CORED.ST',	
       'COREPREF.ST',	'CREDA.ST',	'CTM.ST',	'CTT.ST',	'DIOS.ST',	'DUNI.ST',	'DUST.ST',	'EAST.ST',	'ELANB.ST',	'ELTEL.ST',	'ENEA.ST',	'ENQ.ST',	'EOLUB.ST',
       'FAG.ST',	'FG.ST',	'FINGB.ST',	'FNM.ST',	'G5EN.ST',	'GARO.ST',	'GPG.ST',	'GRNG.ST',	'HEBAB.ST',	'HLDX.ST',	'HMS.ST',	'HNSA.ST',	'HOFI.ST',
       'HTRO.ST',	'HUM.ST',	'IARB.ST',	'IBTB.ST',	'IMMNOV.ST',	'INSTAL.ST',	'INWI.ST',	'IPCO.ST',	'IRLABA.ST',	'ITAB.ST',	'IVSO.ST',	'JOMA.ST',	
       'K2AB.ST',	'K2APREF.ST',	'KAR.ST',	'KARO.ST',	'KFASTB.ST',	'KNOW.ST',	'LAGRB.ST',	'LEO.ST',	'LIAB.ST',	'LIME.ST',	'LINC.ST',	'LUC.ST',	'LUG.ST',	
       'MCAP.ST',	'MEKO.ST',	'MIPS.ST',	'MMGRB.ST',	'MSONA.ST',	'MSONB.ST',	'MTGA.ST',	'MTGB.ST',	'MTRS.ST',	'NCAB.ST',	'NEWAB.ST',	'NMAN.ST',	'NOBINA.ST',
       'NP3.ST',	'NP3PREF.ST',	'NPAPER.ST',	'NWG.ST',	'OASM.ST',	'OEMB.ST',	'ONCO.ST',	'ORES.ST',	'ORX.ST',	'OVZON.ST',	'PACT.ST',	'PIERCE.ST',	'PRFO.ST',
       'PRICB.ST',	'PROB.ST',	'QLINEA.ST',	'RAYB.ST',	'READ.ST',	'REJLB.ST',	'RROS.ST',	'RVRC.ST',	'SAS.ST',	'SCST.ST',	'SHOT.ST',	'SIVE.ST',	'SKISB.ST',
       'STEFB.ST',	'SVOLA.ST',	'SVOLB.ST',	'SYSR.ST',	'TETY.ST',	'TFBANK.ST',	'TOBII.ST',	'TRACB.ST',	'TRIANB.ST',	'TROAX.ST',	'VBGB.ST',	'VITB.ST',	
       'VNV.ST',	'VOLO.ST',	'VOLOPREF.ST',	'WBGRB.ST',	'XANOB.ST',	'XSPRAY.ST',	'XVIVO.ST',	'ABLI.ST',	'ACE.ST',	'ACTI.ST',	'ANOT.ST',	'ARISE.ST',	
       'ARP.ST',	'ARPL.ST',	'ATIC.ST',	'ATORX.ST',	'ATVEXAB.ST',	'B3.ST',	'BEGR.ST',	'BELE.ST',	'BINV.ST',	'BONG.ST',	'BORG.ST',	'BOUL.ST',	'BRGB.ST',
       'CBTTB.ST',	'CCORB.ST',	'CNCJOB.ST',	'CRADB.ST',	'DEDI.ST',	'DORO.ST',	'DURCB.ST',	'EGTX.ST',	'ELEC.ST',	'ELOSB.ST',	'EMPIRB.ST',	'ENDO.ST',
       'ENRO.ST',	'ENROPREFA.ST',	'ENROPREFB.ST',	'EPISB.ST',	'ETX.ST',	'EWRK.ST',	'FEEL.ST',	'FMMB.ST',	'FPIP.ST',	'GHP.ST',	'GIGSEK.ST',	'GREEN.ST',	
       'HANZA.ST',	'HAVB.ST',	'IMMU.ST',	'INFREA.ST',	'IRRAS.ST',	'IS.ST',	'JOSE.ST',	'KABEB.ST',	'KDEV.ST',	'LAMMB.ST',	'MAG.ST',	'MAHAA.ST',	'MEABB.ST',
       'MIDWA.ST',	'MIDWB.ST',	'MILDEF.ST',	'MOB.ST',	'MOMENT.ST',	'MSABB.ST',	'MULQ.ST',	'MVIRB.ST',	'NAXS.ST',	'NELLY.ST',	'NETIB.ST',	'NGS.ST',
       'NILB.ST',	'NOTE.ST',	'NTEKB.ST',	'ODD.ST',	'OP.ST',	'OPPREF.ST',	'OPPREFB.ST',	'ORTIA.ST',	'ORTIB.ST',	'PENGB.ST',	'POOLB.ST',	'PREC.ST',	'PREVB.ST',	
       'PROFB.ST',	'QLIRO.ST',	'RAIL.ST',	'RIZZOB.ST',	'RNBS.ST',	'SANION.ST',	'SEMC.ST',	'SENS.ST',	'SEZI.ST',	'SINT.ST',	'SLEEP.ST',	'SOFB.ST',
       'SRNKEB.ST',	'STARA.ST',	'STARB.ST',	'STRAX.ST',	'STWK.ST',	'SVEDB.ST',	'SVIK.ST',	'TRAD.ST',	'VICO.ST',	'VSSABB.ST',	'WISE.ST',	'XBRANE.ST',	
       'ZETA.ST',]]]'''
          
#print(Symbols)
# create empty dataframe
stock_final_v2 = pd.DataFrame()

t0 = time.time()

# create empty dataframe
stock_final_v2 = pd.DataFrame()

# iterate over each symbol
for i in Symbols:

    # print the symbol which is being downloaded
    print( str(Symbols.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)

    try:
        # download the stock price
        stock = []
        stock = yf.download(i,start=start, end=end, progress=False)

        # append the individual stock prices
        if len(stock) == 0:
            None
        else:
            stock['Name']=i
            stock_final_v2 = stock_final_v2.append(stock,sort=False)
    except Exception:
        None

t1 = time.time()

total = t1-t0

#print(stock_final.head())
print(len(stock_final_v2))
print(stock_final_v2.head(10))
print(stock_final_v2.tail(10))


#stock_final_v2.to_csv('/Users/joeriksson/Desktop/python_data/stock_swe_2021401.csv')
