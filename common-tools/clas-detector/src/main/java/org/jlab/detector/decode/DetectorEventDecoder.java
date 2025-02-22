package org.jlab.detector.decode;

import java.util.Arrays;
import java.util.List;
import org.jlab.detector.base.DetectorType;
import org.jlab.detector.calib.utils.ConstantsManager;
import org.jlab.detector.decode.DetectorDataDgtz.ADCData;
import org.jlab.utils.groups.IndexedTable;

/**
 *
 * @author gavalian
 */
public class DetectorEventDecoder {

    ConstantsManager translationManager = new ConstantsManager();
    ConstantsManager fitterManager      = new ConstantsManager();
    ConstantsManager scalerManager      = new ConstantsManager();

    List<String> tablesTrans  = null;
    List<String> keysTrans    = null;
    List<String> tablesFitter = null;
    List<String> keysFitter   = null;

    private int runNumber = 10;

    private ExtendedFADCFitter extendedFitter = new ExtendedFADCFitter();
    private MVTFitter mvtFitter = new MVTFitter();

    public DetectorEventDecoder(boolean development){
        if(development==true){
            this.initDecoderDev();
        } else {
            this.initDecoder();
        }
    }

    public void setTimestamp(String timestamp) {
        translationManager.setTimeStamp(timestamp);
        fitterManager.setTimeStamp(timestamp);
        scalerManager.setTimeStamp(timestamp);
    }

    public void setVariation(String variation) {
        translationManager.setVariation(variation);
        fitterManager.setVariation(variation);
        scalerManager.setVariation(variation);
    }

    public void setRunNumber(int run){
        this.runNumber = run;
    }

    public int getRunNumber() {
        return this.runNumber;
    }

    public float getRcdbTorusScale() {
        return ((Double)this.scalerManager.getRcdbConstant(this.runNumber,"torus_scale").
                getValue()).floatValue();
    }

    public float getRcdbSolenoidScale() {
        return ((Double)this.scalerManager.getRcdbConstant(this.runNumber,"solenoid_scale").
                getValue()).floatValue();
    }

    public DetectorEventDecoder(){
        this.initDecoder();
    }

    public final void initDecoderDev(){
        keysTrans = Arrays.asList(new String[]{ "HTCC","BST","RTPC"} );
        tablesTrans = Arrays.asList(new String[]{ "/daq/tt/clasdev/htcc","/daq/tt/clasdev/svt","/daq/tt/clasdev/rtpc" });
        keysFitter   = Arrays.asList(new String[]{"HTCC"});
        tablesFitter = Arrays.asList(new String[]{"/daq/fadc/clasdev/htcc"});
        translationManager.init(keysTrans,tablesTrans);
        fitterManager.init(keysFitter, tablesFitter);
        scalerManager.init(Arrays.asList(new String[]{"/runcontrol/fcup","/runcontrol/slm","/runcontrol/hwp",
                                                      "/runcontrol/helicity","/daq/config/scalers/dsc1"}));
    }

    public final void initDecoder(){

        // Detector translation table
        keysTrans = Arrays.asList(new String[]{"FTCAL","FTHODO","FTTRK","LTCC","ECAL","FTOF",
                                               "HTCC","DC","CTOF","CND","BST","RF","BMT","FMT",
                                               "RICH","HEL","BAND","RTPC",
                                               "RASTER","ATOF","AHDC"
        });
        tablesTrans = Arrays.asList(new String[]{
            "/daq/tt/ftcal","/daq/tt/fthodo","/daq/tt/fttrk","/daq/tt/ltcc",
            "/daq/tt/ec","/daq/tt/ftof","/daq/tt/htcc","/daq/tt/dc","/daq/tt/ctof","/daq/tt/cnd","/daq/tt/svt",
            "/daq/tt/rf","/daq/tt/bmt","/daq/tt/fmt","/daq/tt/rich2","/daq/tt/hel","/daq/tt/band","/daq/tt/rtpc",
            "/daq/tt/raster","/daq/tt/atof","/daq/tt/ahdc"
        });
        translationManager.init(keysTrans,tablesTrans);
        
        // ADC waveform fitter translation table
        keysFitter   = Arrays.asList(new String[]{"FTCAL","FTHODO","FTTRK","FTOF","LTCC",
                                                  "ECAL","HTCC","CTOF","CND","BMT",
                                                  "FMT","HEL","RF","BAND","RASTER",
                                                  "AHDC"});
        tablesFitter = Arrays.asList(new String[]{
            "/daq/fadc/ftcal","/daq/fadc/fthodo","/daq/config/fttrk","/daq/fadc/ftof","/daq/fadc/ltcc",
            "/daq/fadc/ec", "/daq/fadc/htcc","/daq/fadc/ctof","/daq/fadc/cnd","/daq/config/bmt",
            "/daq/config/fmt","/daq/fadc/hel","/daq/fadc/rf","/daq/fadc/band","/daq/fadc/raster",
            "/daq/config/ahdc"
        });
        fitterManager.init(keysFitter, tablesFitter);

        scalerManager.init(Arrays.asList(new String[]{"/runcontrol/fcup","/runcontrol/slm","/runcontrol/hwp",
                                                      "/runcontrol/helicity","/daq/config/scalers/dsc1"}));
    }

    /**
     * applies translation table to the digitized data to translate
     * crate,slot channel to sector layer component.
     * @param detectorData
     */
    public void translate(List<DetectorDataDgtz>  detectorData){

        for(DetectorDataDgtz data : detectorData){

            int crate    = data.getDescriptor().getCrate();
            int slot     = data.getDescriptor().getSlot();
            int channel  = data.getDescriptor().getChannel();

            for(String table : keysTrans){
                IndexedTable  tt = translationManager.getConstants(runNumber, table);
                DetectorType  type = DetectorType.getType(table);
                if(tt.hasEntry(crate,slot,channel)==true){
                    int sector    = tt.getIntValue("sector", crate,slot,channel);
                    int layer     = tt.getIntValue("layer", crate,slot,channel);
                    int component = tt.getIntValue("component", crate,slot,channel);
                    int order     = tt.getIntValue("order", crate,slot,channel);

                    data.getDescriptor().setSectorLayerComponent(sector, layer, component);
                    data.getDescriptor().setOrder(order);
                    data.getDescriptor().setType(type);

                    for(int i = 0; i < data.getADCSize(); i++) {
                        data.getADCData(i).setOrder(order);
                    }
                    for(int i = 0; i < data.getTDCSize(); i++) {
                        data.getTDCData(i).setOrder(order);
                    }
                }
            }
        }
    }

    public void fitPulses(List<DetectorDataDgtz>  detectorData){
        for(DetectorDataDgtz data : detectorData){
            int crate    = data.getDescriptor().getCrate();
            int slot     = data.getDescriptor().getSlot();
            int channel  = data.getDescriptor().getChannel();
            for(String table : keysFitter){
                //custom MM fitter
            	if( ( (table.equals("BMT"))&&(data.getDescriptor().getType().getName().equals("BMT")) )
                 || ( (table.equals("FMT"))&&(data.getDescriptor().getType().getName().equals("FMT")) )
                 || ( (table.equals("AHDC"))&&(data.getDescriptor().getType().getName().equals("AHDC")) )
                 || ( (table.equals("FTTRK"))&&(data.getDescriptor().getType().getName().equals("FTTRK")) ) ){
                    IndexedTable daq = fitterManager.getConstants(runNumber, table);
                    short adcOffset = (short) daq.getDoubleValue("adc_offset", 0, 0, 0);
                    double fineTimeStampResolution = (byte) daq.getDoubleValue("dream_clock", 0, 0, 0);
                    double samplingTime = (byte) daq.getDoubleValue("sampling_time", 0, 0, 0);
                    int sparseSample = daq.getIntValue("sparse", 0, 0 ,0);
                    if (data.getADCSize() > 0) {
                        ADCData adc = data.getADCData(0);
                        mvtFitter.fit(adcOffset, fineTimeStampResolution, samplingTime, adc.getPulseArray(), adc.getTimeStamp(), sparseSample);
                        adc.setHeight((short) (mvtFitter.adcMax));
                        adc.setTime((int) (mvtFitter.timeMax));
                        adc.setIntegral((int) (mvtFitter.integral));
                        adc.setTimeStamp(mvtFitter.timestamp);
                    }
                } else {
                    IndexedTable  daq = fitterManager.getConstants(runNumber, table);
                    DetectorType  type = DetectorType.getType(table);
                    if(daq.hasEntry(crate,slot,channel)==true){
                        int nsa = daq.getIntValue("nsa", crate,slot,channel);
                        int nsb = daq.getIntValue("nsb", crate,slot,channel);
                        int tet = daq.getIntValue("tet", crate,slot,channel);
                        int ped = 0;
                        if(table.equals("RF")&&data.getDescriptor().getType().getName().equals("RF")) ped = daq.getIntValue("pedestal", crate,slot,channel);
                        if(data.getADCSize()>0){
                            for(int i = 0; i < data.getADCSize(); i++){
                                ADCData adc = data.getADCData(i);
                                if(adc.getPulseSize()>0){
                                    try {
                                        extendedFitter.fit(nsa, nsb, tet, ped, adc.getPulseArray());
                                    } catch (Exception e) {
                                        System.out.println(">>>> error : fitting pulse "
                                                            +  crate + " / " + slot + " / " + channel);
                                    }
                                    int adc_corrected = extendedFitter.adc + extendedFitter.ped*(nsa+nsb);
                                    adc.setHeight((short) this.extendedFitter.pulsePeakValue);
                                    adc.setIntegral(adc_corrected);
                                    adc.setTimeWord(this.extendedFitter.t0);
                                    adc.setPedestal((short) this.extendedFitter.ped);
                                }
                            }
                        }
                        if(data.getADCSize()>0){
                            for(int i = 0; i < data.getADCSize(); i++){
                                    data.getADCData(i).setADC(nsa, nsb);
                            }
                        }
                    }
                }
            }
        }
    }
}
