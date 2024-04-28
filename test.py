import re
import inspect

from utils import Profile

profile = Profile(profile={"xyz":True})

print(profile.get())
profile.dump(save_to="test.yaml")