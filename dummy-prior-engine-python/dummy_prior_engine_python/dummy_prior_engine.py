class dummy_prior_engine:

    def create_prior(self, aux_data):
        print 'I create a prior'
        return []

    def save_prior(self, prior):
        print 'I save a prior'

    def get_prior(self, prior_id):
        print 'I return a prior'
        return []