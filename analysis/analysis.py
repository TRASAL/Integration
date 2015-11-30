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

def statistics(queue, table, samples):
    confs = list()
    dms_range = manage.get_dm_range(queue, table, samples)
    for dm in dms_range:
        queue.execute("SELECT MIN(GFLOPs),AVG(GFLOPs),MAX(GFLOPs),STDDEV_POP(GFLOPs),MIN(GBs),AVG(GBs),MAX(GBs),STDDEV_POP(GBs) FROM " + table + " WHERE (DMs = " + str(dm[0]) + " AND samples = " + samples + ")")
        line = queue.fetchall()
        confs.append([dm[0], line[0][0], line[0][2], line[0][1], line[0][3], (line[0][2] - line[0][1]) / line[0][3], line[0][4], line[0][5], line[0][6], line[0][7], (line[0][6] - line[0][5]) / line[0][7]])
    return confs

def histogram(queue, table, samples):
    hists = list()
    dms_range = manage.get_dm_range(queue, table, samples)
    for dm in dms_range:
        queue.execute("SELECT MAX(GBs) FROM " + table + " WHERE (DMs = " + str(dm[0]) + " AND samples = " + samples + ")")
        maximum = int(queue.fetchall()[0][0])
        hist = [0 for i in range(0, maximum + 1)]
        queue.execute("SELECT GBs FROM " + table + " WHERE (DMs = " + str(dm[0]) + " AND samples = " + samples + ")")
        flops = queue.fetchall()
        for flop in flops:
            hist[int(flop[0])] = hist[int(flop[0])] + 1
        hists.append(hist)
    return hists

def optimization_space(queue, table, samples):
    confs = list()
    dms_range = manage.get_dm_range(queue, table, samples)
    for dm in dms_range:
        queue.execute("SELECT nrThreadsD0,nrThreadsD1,nrItemsD0,nrItemsD1,GBS FROM " + table + " WHERE (DMs = " + str(dm[0]) + " AND samples = " + samples + ")")
        best = queue.fetchall()
        confs.append([best[0][0], best[0][1], best[0][2], best[0][3], best[0][4], best[0][5]])
    return confs

def single_parameter_space(queue, table, parameter, samples):
    confs = list()
    scenario = "(samples = " + samples + ")"
    dms_range = manage.get_dm_range(queue, table, samples)
    for dm in dms_range:
        internalConf = list()
        queue.execute("SELECT DISTINCT " + parameter + " FROM " + table + " WHERE DMs = " + str(dm[0]) + " AND " + scenario + " ORDER BY " + parameter)
        values = queue.fetchall()
        for value in values:
            queue.execute("SELECT MAX(GFLOPs) FROM " + table + " WHERE " + parameter + " = " + str(value[0]) + " AND DMs = " + str(dm[0]) + " AND " + scenario)
            best = queue.fetchall()
            internalConf.append([value[0], best[0][0]])
        confs.append(internalConf)
    return confs

