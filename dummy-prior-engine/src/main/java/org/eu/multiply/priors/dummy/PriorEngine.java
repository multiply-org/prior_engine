package org.eu.multiply.priors.dummy;

import org.eu.multiply.auxdata.dummy.AuxData;

/**
 * @author Tonio Fincke
 */
public interface PriorEngine {

    Prior createPrior(AuxData auxData);

    Prior getPrior(String name);

    void savePrior(Prior prior);

}
