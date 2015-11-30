#!/usr/bin/env python
# Copyright 2015 Alessio Sclocco <a.sclocco@vu.nl>
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

import manage

def tune(queue, table, operator, samples):
    confs = list()
    if operator.casefold() == "max" or operator.casefold() == "min":
        dms_range = manage.get_dm_range(queue, table, samples)
        for dm in dms_range:
            queue.execute("SELECT nrThreadsD0,nrItemsD0,GFLOPs,GBs,time,time_err,cov FROM " + table + " WHERE (GFLOPs = (SELECT " + operator + "(GFLOPs) FROM " + table + " WHERE (DMs = " + str(dm[0]) + " AND samples = " + samples + "))) AND (DMs = " + str(dm[0]) + " AND samples = " + samples + ")")
            best = queue.fetchall()
            confs.append([dm[0], best[0][0], best[0][1], best[0][2], best[0][3], best[0][4], best[0][5], best[0][6]])
    return confs

