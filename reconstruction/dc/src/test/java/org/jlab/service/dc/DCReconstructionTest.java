package org.jlab.service.dc;

import cnuphys.magfield.MagneticFields;
import org.junit.Test;
import static org.junit.Assert.*;

import org.jlab.io.base.DataEvent;

import org.jlab.analysis.physics.TestEvent;
import org.jlab.analysis.math.ClasMath;

import org.jlab.clas.swimtools.MagFieldsEngine;
import org.jlab.detector.base.DetectorType;
import org.jlab.logging.DefaultLogger;
import org.jlab.utils.CLASResources;

/**
 *
 * @author nharrison, marmstr
 */
public class DCReconstructionTest {

  @Test
  public void testDCReconstruction() {

    DefaultLogger.debug();

    System.setProperty("CLAS12DIR", "../../");

    String mapDir = CLASResources.getResourcePath("etc")+"/data/magfield";
    try {
        MagneticFields.getInstance().initializeMagneticFields(mapDir,
                "Symm_torus_r2501_phi16_z251_24Apr2018.dat","Symm_solenoid_r601_phi1_z1201_13June2018.dat");
    }
    catch (Exception e) {
        e.printStackTrace();
    }

    DataEvent testEvent = TestEvent.get(DetectorType.DC);

    MagFieldsEngine enf = new MagFieldsEngine();
    enf.init();
    enf.processDataEvent(testEvent);
    DCHBClustering      engineCL = new DCHBClustering();
    DCHBPostClusterConv engineHB = new DCHBPostClusterConv();
    engineCL.init();
    engineHB.init();
    engineCL.processDataEvent(testEvent); 
    engineHB.processDataEvent(testEvent); 
    if(testEvent.hasBank("HitBasedTrkg::HBTracks")) {
        testEvent.getBank("HitBasedTrkg::HBTracks").show();
    }
    
    //Compare HB momentum to expectation
    assertEquals(testEvent.hasBank("HitBasedTrkg::HBTracks"), true);
    assertEquals(testEvent.getBank("HitBasedTrkg::HBTracks").rows(), 1);
    assertEquals(testEvent.getBank("HitBasedTrkg::HBTracks").getByte("q", 0), -1);
    assertEquals(ClasMath.isWithinXPercent(16.0, testEvent.getBank("HitBasedTrkg::HBTracks").getFloat("p0_x", 0), 1.057), true);
    assertEquals(testEvent.getBank("HitBasedTrkg::HBTracks").getFloat("p0_y", 0) > -0.1, true);
    assertEquals(testEvent.getBank("HitBasedTrkg::HBTracks").getFloat("p0_y", 0) < 0.1, true);
    assertEquals(ClasMath.isWithinXPercent(16.0, testEvent.getBank("HitBasedTrkg::HBTracks").getFloat("p0_z", 0), 2.266), true);

    //TB reconstruction
    DCTBEngine engineTB = new DCTBEngine();
    engineTB.init();
    engineTB.processDataEvent(testEvent); 
    if(testEvent.hasBank("TimeBasedTrkg::TBTracks")) {
        testEvent.getBank("TimeBasedTrkg::TBTracks").show();
    }
    
    assertEquals(testEvent.hasBank("TimeBasedTrkg::TBTracks"), true);
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBTracks").rows(), 1);
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBTracks").getByte("q", 0), -1);

    assertEquals(ClasMath.isWithinXPercent(27.9, testEvent.getBank("TimeBasedTrkg::TBTracks").getFloat("p0_x", 0), 0.997), true);
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBTracks").getFloat("p0_y", 0) > -0.0702, true);
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBTracks").getFloat("p0_y", 0) < 0.0438, true);
    assertEquals(ClasMath.isWithinXPercent(17.5, testEvent.getBank("TimeBasedTrkg::TBTracks").getFloat("p0_z", 0), 2.04), true);

    assertEquals(testEvent.getBank("TimeBasedTrkg::TBTracks").getFloat("Vtx0_x", 0) < 0.2, true);
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBTracks").getFloat("Vtx0_x", 0) > -0.2, true);
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBTracks").getFloat("Vtx0_y", 0) < 0.5, true);  
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBTracks").getFloat("Vtx0_y", 0) > -0.5, true);
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBTracks").getFloat("Vtx0_z", 0) < 0.885, true);
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBTracks").getFloat("Vtx0_z", 0) > -0.0753, true);
    
    //Region 1
    assertEquals(ClasMath.isWithinXPercent(155, testEvent.getBank("TimeBasedTrkg::TBCrosses").getFloat("x", 0), 4.02), true); 
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBCrosses").getFloat("y", 0 ) < 9.25, true); 
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBCrosses").getFloat("y", 0 ) > -11.78, true); 
 
    //Region 2 
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBCrosses").getFloat("x", 1 ) < 14.2, true); 
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBCrosses").getFloat("x", 1 ) > -5.8, true);
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBCrosses").getFloat("y", 1 ) < 13.9, true); 
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBCrosses").getFloat("y", 1 ) > -17.8, true); 
    
    //Region 3
    assertEquals(ClasMath.isWithinXPercent(127, testEvent.getBank("TimeBasedTrkg::TBCrosses").getFloat("x", 2), -11.0), true); 
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBCrosses").getFloat("y", 2 ) < 17.96, true); 
    assertEquals(testEvent.getBank("TimeBasedTrkg::TBCrosses").getFloat("y", 2 ) > -23.66, true); 
    
  }
  
}
