
class InteractorCache:
    def get_id(self,dm,evaluation_policy,itr):
        return dm.get_id()+'/'+evaluation_policy.get_id()+'/'+itr.get_id()
