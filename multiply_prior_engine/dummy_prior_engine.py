class DummyPriorEngine:

    def create_prior(self, aux_data):
        print 'Creating a prior'
        return []

    def save_prior(self, prior):
        print 'Saving a prior'

    def get_prior(self, prior_id):
        print 'Returning a prior'
        return []