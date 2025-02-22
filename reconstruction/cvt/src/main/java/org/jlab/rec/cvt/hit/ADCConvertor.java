package org.jlab.rec.cvt.hit;

import java.util.Random;

import org.jlab.rec.cvt.Constants;
import org.jlab.rec.cvt.svt.SVTParameters;

public class ADCConvertor {

    public ADCConvertor() {

    }

    public static boolean isEventUnCorrupted(int adc, int adcstat) {
        //The value adcStatus is 0 for runs where adc=-1 is permitted and 1 for runs where adc=-1 is NOT permitted
        //
        boolean pass = true;
        if(adc==-1) {
            if(adcstat==1) //1: event corrupted; 0 event is OK
                pass=false;
            if(adcstat==0) 
                pass=true;
        }
        return pass;
    }
     /**
     *
     * @param adc ADC value Converts ADC values to DAQ units -- used for BST
     * test stand analysis
     * @return 
     */
    public static double SVTADCtoDAQ(int adc) {
        
        if (adc < 0 || adc > 7) {
            return -1;
        }
        
        int START[] = new int[8];
        int END[] = new int[8];
        for (int i = 0; i < 8; i++) {
            START[i] = SVTParameters.INITTHRESHOLD + SVTParameters.DELTATHRESHOLD * i;
            END[i] = SVTParameters.INITTHRESHOLD + SVTParameters.DELTATHRESHOLD * (i + 1);
        }
        END[7] = 1000; //overflow

        Random random = new Random();
        random.setSeed(42);

        //int daq = returnRandomInteger(START[adc], END[adc], random);
        //double value = (double) daq;
        double value = (double) 0.5*(END[adc]+START[adc]);
        
        return value;
    }

    private static int returnRandomInteger(int aStart, int aEnd, Random aRandom) {
        if (aStart > aEnd) {
            return 0;
            //throw new IllegalArgumentException("Start cannot exceed End.");
        }
        /*
        //get the range, casting to long to avoid overflow problems -- for a flat distribution
        long range = (long)aEnd - (long)aStart + 1;
        // compute a fraction of the range, 0 <= frac < range
        double x = aRandom.nextDouble();
        long fraction = (long)(range * x);
        int randomNumber =  (int)(fraction + aStart);  
         */
        double landauC = -1;
        while (landauC > aEnd || landauC < aStart) {
            double landau = randomLandau(100, 5, aRandom);
            landauC = (41 * aRandom.nextGaussian() + landau);
        }

        int randomNumber = (int) landauC;

        return randomNumber;
    }

////////////////////////////////////////////////////////////////////////////////
    /// Generate a random number following a Landau distribution
    /// The Landau random number generation is implemented using the
    /// function landau_quantile(x,sigma), which provides
    /// the inverse of the landau cumulative distribution.
    /// landau_quantile has been converted from CERNLIB ranlan(G110).
    private static double randomLandau(double mu, double sigma, Random aRandom) {

        if (sigma <= 0) {
            return 0;
        }

        double res = mu + landauQuantile(aRandom.nextDouble(), sigma);
        return res;
    }

    private static double landauQuantile(double z, double xi) {
        // LANDAU quantile : algorithm from CERNLIB G110 ranlan
        // with scale parameter xi

        if (xi <= 0) {
            return 0;
        }
        if (z <= 0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (z >= 1) {
            return Double.POSITIVE_INFINITY;
        }

        double ranlan, u, v;
        u = 1000 * z;
        int i = (int) u;
        u -= i;
        if (i >= 70 && i < 800) {
            ranlan = Constants.f[i - 1] + u * (Constants.f[i] - Constants.f[i - 1]);
        } else if (i >= 7 && i <= 980) {
            ranlan = Constants.f[i - 1] + u * (Constants.f[i] - Constants.f[i - 1] - 0.25 * (1 - u) * (Constants.f[i + 1] - Constants.f[i] - Constants.f[i - 1] + Constants.f[i - 2]));
        } else if (i < 7) {
            v = Math.log(z);
            u = 1 / v;
            ranlan = ((0.99858950 + (3.45213058E1 + 1.70854528E1 * u) * u)
                    / (1 + (3.41760202E1 + 4.01244582 * u) * u))
                    * (-Math.log(-0.91893853 - v) - 1);
        } else {
            u = 1 - z;
            v = u * u;
            if (z <= 0.999) {
                ranlan = (1.00060006 + 2.63991156E2 * u + 4.37320068E3 * v)
                        / ((1 + 2.57368075E2 * u + 3.41448018E3 * v) * u);
            } else {
                ranlan = (1.00001538 + 6.07514119E3 * u + 7.34266409E5 * v)
                        / ((1 + 6.06511919E3 * u + 6.94021044E5 * v) * u);
            }
        }
        return xi * ranlan;
    }

}
