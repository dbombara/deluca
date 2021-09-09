# Copyright 2021 The Deluca Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Residual controller."""
import deluca.core
from deluca.lung import core
import jax


class CompositeState(deluca.Obj):
  """Combined state of two controller states."""
  base_state: deluca.Obj
  resid_state: deluca.Obj

  @property
  def time(self):
    return self.base_state.time

  @property
  def steps(self):
    return self.base_state.steps

  @property
  def dt(self):
    return self.base_state.dt


class CompositeController(core.Controller):
  """Controller comprised of a base controller along with a residual controller.

  Only the residual controller is trainable.
  """
  base_controller: core.Controller = deluca.field(jaxed=False)
  resid_controller: core.Controller = deluca.field(jaxed=True)

  def init(self, waveform=None):
    return CompositeState.create(
        self.base_controller.init(waveform),
        self.resid_controller.init(waveform))

  @jax.jit
  def __call__(self, controller_state: CompositeState, obs):
    base_state, u_in_base = self.base_controller(controller_state.base_state,
                                                 obs)
    resid_state, u_in_resid = self.resid_controller(
        controller_state.resid_state, obs)
    u_in = u_in_base + u_in_resid

    return CompositeState.create(base_state, resid_state), u_in
