package org.eu.multiply.priors.dummy;

import org.eu.multiply.auxdata.dummy.AuxData;

/**
 * @author Tonio Fincke
 */
public class DummyPriorEngine implements PriorEngine {

    public Prior createPrior(AuxData auxData) {
        return new DummyPrior();
    }

    public Prior getPrior(String name) {
        return new DummyPrior();
    }

    public void savePrior(Prior prior) {
        // do nothing
    }

}
