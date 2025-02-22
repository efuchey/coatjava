package org.jlab.rec.cvt.banks;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.jlab.clas.swimtools.Swim;
import org.jlab.detector.banks.RawDataBank;
import org.jlab.detector.base.DetectorDescriptor;
import org.jlab.detector.base.DetectorType;
import org.jlab.io.base.DataEvent;
import org.jlab.rec.cvt.Constants;
import org.jlab.rec.cvt.Geometry;
import org.jlab.rec.cvt.bmt.BMTGeometry;
import org.jlab.rec.cvt.bmt.BMTType;
import org.jlab.rec.cvt.hit.ADCConvertor;
import org.jlab.rec.cvt.hit.Hit;
import org.jlab.rec.cvt.hit.Strip;
import org.jlab.rec.cvt.svt.SVTGeometry;
import org.jlab.utils.groups.IndexedTable;

/**
 * A class to fill in lists of hits corresponding to reconstructed hits
 * characterized by the strip, its location in the detector (layer, sector), its
 * reconstructed time.
 *
 * @author ziegler
 *
 */
public class HitReader {

    public HitReader() {
        
    }

    // the list of BMT hits
    private List<Hit> _BMTHits;

    /**
     *
     * @return a list of BMT hits
     */
    public List<Hit> getBMTHits() {
        return _BMTHits;
    }

    /**
     * sets the list of BMT hits
     *
     * @param _BMTHits list of BMT hits
     */
    public void setBMTHits(List<Hit> _BMTHits) {
        this._BMTHits = _BMTHits;
    }
    // the list of SVT hits
    private List<Hit> _SVTHits;

    /**
     *
     * @return a list of SVT hits
     */
    public List<Hit> getSVTHits() {
        return _SVTHits;
    }

    /**
     * sets the list of SVT hits
     *
     * @param _SVTHits list of SVT hits
     */
    public void setSVTHits(List<Hit> _SVTHits) {
        this._SVTHits = _SVTHits;
    }
    
    /**
     * Gets the BMT hits from the BMT dgtz bank
     *
     * @param event the data event
     * @param swim
     * @param status
     * @param timeCuts
     */
    public void fetch_BMTHits(DataEvent event, Swim swim, IndexedTable status, 
            IndexedTable timeCuts, IndexedTable bmtStripVoltage, IndexedTable bmtStripVoltageThresh) {

        // return if there is no BMT bank
        if (event.hasBank("BMT::adc") == false) {
            //System.err.println("there is no BMT bank ");
            _BMTHits = new ArrayList<>();

            return;
        }
        
        // instanciates the list of hits
        List<Hit> hits = new ArrayList<>();
        List<Hit> hits50c = new ArrayList<>();
        List<Hit> hits50z = new ArrayList<>();
        // gets the BMT dgtz bank
        RawDataBank bankDGTZ = new RawDataBank("BMT::adc");
        bankDGTZ.read(event);
        // fills the arrays corresponding to the hit variables
        int rows = bankDGTZ.rows();

        if (event.hasBank("BMT::adc") == true) {
            
            double tmin = timeCuts.getDoubleValue("hit_min", 0,0,0);
            double tmax = timeCuts.getDoubleValue("hit_max", 0,0,0);
            
            for (int i = 0; i < rows; i++) {

                //if (bankDGTZ.getInt("ADC", i) < 1) {
                    //continue; // gemc assigns strip value -1 for inefficiencies, we only consider strips with values between 1 to the maximum strip number for a given detector
                //}                
                int sector  = bankDGTZ.getByte("sector", i);
                int layer   = bankDGTZ.getByte("layer", i);
                int strip   = bankDGTZ.getShort("component", i);
                double ADCtoEdep = bankDGTZ.getInt("ADC", i);
                double time = bankDGTZ.getFloat("time", i);
                int order   = bankDGTZ.trueOrder(i);
                //if (order == 1) {
                //    continue;
                //}
                //fix for now... no adc in GEMC
                if(Constants.getInstance().gemcIgnBMT0ADC==false) {
                    if (ADCtoEdep < 1) {
                        continue;
                    }
                }
                if(strip<1) {
                    continue;
                }
                // create the strip object for the BMT
                Strip BmtStrip = new Strip(strip, ADCtoEdep, time);
                BmtStrip.setStatus(status.getIntValue("status", sector, layer, strip));
                if(Constants.getInstance().timeCuts) {
                    if(time!=0 && (time<tmin || time>tmax))
                        BmtStrip.setStatus(2);// calculate the strip parameters for the BMT hit
                }
                if(Constants.getInstance().bmtHVCuts) {
                    if(bmtStripVoltage!=null && bmtStripVoltage.hasEntry(sector,layer,0) && 
                            bmtStripVoltageThresh!=null && bmtStripVoltageThresh.hasEntry(sector,layer,0)) {
                        double hv  = bmtStripVoltage.getDoubleValue("HV", sector,layer,0); 
                        double hv1 = bmtStripVoltageThresh.getDoubleValue("HV1", sector,layer,0); 
                        double hv2 = bmtStripVoltageThresh.getDoubleValue("HV2", sector,layer,0); 
                        double hv3 = bmtStripVoltageThresh.getDoubleValue("HV3", sector,layer,0); 
                        
                        if(hv<hv1) 
                            BmtStrip.setStatus(4);
                        if(hv>=hv1 && hv<hv2) 
                            BmtStrip.setStatus(5);
                        if(hv>=hv2 && hv<hv3) 
                            BmtStrip.setStatus(6);
                    }
                }
                BmtStrip.calcBMTStripParams(sector, layer, swim); // for Z detectors the Lorentz angle shifts the strip measurement; calc_Strip corrects for this effect
                // create the hit object for detector type BMT
                
                Hit hit = new Hit(DetectorType.BMT, BMTGeometry.getDetectorType(layer), sector, layer, BmtStrip);                
                hit.setId(bankDGTZ.trueIndex(i)+1);
                if (Constants.getInstance().flagSeeds)
                    hit.MCstatus = order;
                
                // add this hit
                if(hit.getLayer()+3!=Constants.getInstance().getRmReg()) {
                    if(Constants.getInstance().useOnlyMCTruthHits() ) {
                        if(hit.MCstatus==0)
                            hits.add(hit);
                    } 
                    else if(Constants.getInstance().useOnlyBMTTruthHits ) {
                        if(hit.MCstatus==0)
                            hits.add(hit);
                    }
                    else if(Constants.getInstance().useOnlyBMTCTruthHits && hit.getType()==BMTType.C) {
                        if(hit.MCstatus==0)
                            hits.add(hit);
                    }
                    else if(Constants.getInstance().useOnlyBMTZTruthHits && hit.getType()==BMTType.Z) {
                        if(hit.MCstatus==0)
                            hits.add(hit);
                    }
                    else if(Constants.getInstance().useOnlyBMTC50PercTruthHits && hit.getType()==BMTType.C) {
                        if(hit.MCstatus==0)
                            hits.add(hit);
                        if(hit.MCstatus==1)
                            hits50c.add(hit);
                    }
                    else if(Constants.getInstance().useOnlyBMTC50PercTruthHits && hit.getType()==BMTType.Z) {
                        if(hit.MCstatus==0)
                            hits.add(hit);
                        if(hit.MCstatus==1)
                            hits50z.add(hit);
                    }
                    else {
                        hits.add(hit);
                    }
                }
            }
            if(Constants.getInstance().useOnlyBMTC50PercTruthHits) {
                int s = hits50c.size()/2;
                for(int i = 0; i<s; i++) {
                    hits.add(hits50c.get(i));
                }
            }
            if(Constants.getInstance().useOnlyBMTZ50PercTruthHits) {
                int s = hits50z.size()/2;
                for(int i = 0; i<s; i++) {
                    hits.add(hits50z.get(i));
                }
            }
            // fills the list of BMT hits
            Collections.sort(hits);
            
            this.setBMTHits(hits);
        }
    }

    /**
     * Gets the SVT hits from the BMT dgtz bank
     *
     * @param event the data event
     * @param omitLayer
     * @param omitHemisphere
     * @param status
     * @param adcStatus
     */
    public void fetch_SVTHits(DataEvent event, int omitLayer, int omitHemisphere, 
            IndexedTable status, IndexedTable adcStatus) {

        if (event.hasBank("BST::adc") == false) {
            //System.err.println("there is no BST bank ");
            _SVTHits = new ArrayList<>();
            return;
        }

        List<Hit> hits = new ArrayList<>();

        RawDataBank bankDGTZ = new RawDataBank("BST::adc");
        bankDGTZ.read(event);
        int rows = bankDGTZ.rows();
        
        if (event.hasBank("BST::adc") == true) {
            //pass event
            //In RGA Spring 2018 data there should be no BST::adc.ADC=-1 and if found, the event is corrupted. 
            //In which case all the SVT hits are unreliable and they should all be discarted
            //Starting from Fall 2018 all events would have ADC=-1 and this is normal.
            //This ADC=-1 status is in a ccdb table
            //The value adcStatus in ccdb is 1 for runs where ADC=-1 is not permitted and 0 for runs where ADC=-1 is permitted
            
            int adcStat = adcStatus.getIntValue("adcstatus", 0, 0, 0);
            for (int i = 0; i < rows; i++) {     
                int ADC = bankDGTZ.getInt("ADC", i);
                if(ADCConvertor.isEventUnCorrupted(ADC, adcStat)==false) {
                    return;
                }
            }
            
            //bankDGTZ.show();
            // first get tdcs
            Map<Integer, Double> tdcs = new HashMap<>();
            for (int i = 0; i < rows; i++) {                
                if(bankDGTZ.getInt("ADC", i) < 0) {
                    int sector = bankDGTZ.getByte("sector", i);
                    int layer  = bankDGTZ.getByte("layer", i);
                    int strip = bankDGTZ.getShort("component", i);
                    double time = bankDGTZ.getFloat("time", i);
                    
                    //if (order == 1) {
                    //    continue;
                    //}
                    //if(time<SVTParameters.TIMECUTLOW) 
                    //    continue;
                    
                    int key = DetectorDescriptor.generateHashCode(sector, layer, strip);
                    if(tdcs.containsKey(key)) {
                        if(time<tdcs.get(key))
                            tdcs.replace(key, time);
                    }
                    else 
                        tdcs.put(key, time);
                }
            }
                
            // then get real hits
            for (int i = 0; i < rows; i++) {
                if (bankDGTZ.getInt("ADC", i) < 0) {
                    continue; // ignore hits TDC hits with ADC==-1 
                }
                int order   = bankDGTZ.getByte("order", i);
                int id      = i + 1;
                int sector  = bankDGTZ.getByte("sector", i);
                int layer   = bankDGTZ.getByte("layer", i);
                int strip   = bankDGTZ.getShort("component", i);
                int ADC     = bankDGTZ.getInt("ADC", i);
                double time = 0;//bankDGTZ.getFloat("time", i);
                int tdcstrip = 1;
                if(strip>128) tdcstrip = 129;
                int key = DetectorDescriptor.generateHashCode(sector, layer, tdcstrip);
                if(tdcs.containsKey(key)) {
                    time = tdcs.get(key);
                    //time tag
                    if(Constants.getInstance().useSVTTimingCuts) {
                        if(this.passTimingCuts(ADC, time)==false) 
                            continue;
                        }
                }
//                else {
//                    System.out.println("missing time for " + sector + " " + layer + " " + strip);
//                    for(int ii : tdcs.keySet()) {
//                        int s = (ii&0xFF000000)>>24;
//                        int l = (ii&0x00FF0000)>>16;
//                        int c = (ii&0x0000FFFF);
//                        System.out.println("\t"+s+"/"+l+"/"+c);
//                    }
//                    bankDGTZ.show();
//                }
                
                double angle = SVTGeometry.getSectorPhi(layer, sector);
                int hemisphere = (int) Math.signum(Math.sin(angle));
                if (sector == 7 && layer > 6) {
                    hemisphere = 1;
                }
                if (sector == 19 && layer > 6) {
                    hemisphere = -1;
                }
                if (omitHemisphere == -2) {
                    if(layer == omitLayer) {
                        continue;
                    }
                } else {
                    if (hemisphere == omitHemisphere && layer == omitLayer) {
                        continue;
                    }

                }
                // if the strip is out of range skip
                if (strip < 1) {
                    continue;
                }
                if (layer > 6) {
                    continue;
                }
                
                //if(adcConv.SVTADCtoDAQ(ADC[i], event)<50)
                //    continue;
                // create the strip object with the adc value converted to daq value used for cluster-centroid estimate
                
                //boolean isMC = event.hasBank("MC::Particle");
                double E = ADCConvertor.SVTADCtoDAQ(ADC);
                if(E==-1) 
                    continue;
                
                Strip SvtStrip = new Strip(strip, E, time); 
                SvtStrip.setPitch(SVTGeometry.getPitch());
                // get the strip line
                SvtStrip.setLine(Geometry.getInstance().getSVT().getStrip(layer, sector, strip));
                SvtStrip.setModule(Geometry.getInstance().getSVT().getModule(layer, sector));
                SvtStrip.setNormal(Geometry.getInstance().getSVT().getNormal(layer, sector)); 
                // if the hit is useable in the analysis its status is =0
                //if (SvtStrip.getEdep() == 0) {
                //    SvtStrip.setStatus(1);
                //}
                //get status from ccdb
                SvtStrip.setStatus(status.getIntValue("status", sector, layer, strip));
                // create the hit object
                Hit hit = new Hit(DetectorType.BST, BMTType.UNDEFINED, sector, layer, SvtStrip);
                hit.setId(id);
                if (Constants.getInstance().flagSeeds)
                    hit.MCstatus = order;
                // add this hit
                if(hit.getRegion()!=Constants.getInstance().getRmReg()) {     
                    if(Constants.getInstance().useOnlyMCTruthHits() ) {
                        if(hit.MCstatus==0)
                            hits.add(hit);
                    } else {
                        hits.add(hit); 
                    }
                }
            }
        }
        // fill the list of SVT hits
        Collections.sort(hits);
        this.setSVTHits(hits);

    }

    private boolean passTimingCuts(int adc, double time) {
        int tdc = (int) time;
        boolean pass = true;
        if(adc == 0 && ((tdc > 0 && tdc < 160) || tdc > 400)) pass = false;
        else if(adc == 1 && ((tdc > 0 && tdc < 160) || tdc > 340)) pass = false;
        else if(adc == 2 && ((tdc > 0 && tdc < 160) || tdc > 320)) pass = false;
        else if(adc == 3 && ((tdc > 0 && tdc < 160) || tdc > 300)) pass = false;
        else if(adc == 4 && ((tdc > 0 && tdc < 160) || tdc > 290)) pass = false;
        else if(adc == 5 && ((tdc > 0 && tdc < 170) || tdc > 280)) pass = false;
        else if(adc == 6 && ((tdc > 0 && tdc < 180) || tdc > 280)) pass = false;
        else if(adc == 7 && ((tdc > 0 && tdc < 170) || tdc > 280)) pass = false;
    
        return pass;   
    }

}
