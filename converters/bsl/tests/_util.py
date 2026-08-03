# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def evidence_ossie() -> str:
    return (FIXTURES / "self_plan_evidence.ossie.yaml").read_text()


def rules_ossie() -> str:
    return (FIXTURES / "self_plan.rules.ossie.yaml").read_text()


def prescription_evidence_ossie() -> str:
    return (FIXTURES / "self_prescription_evidence.ossie.yaml").read_text()


def prescription_rules_ossie() -> str:
    return (FIXTURES / "self_prescription.rules.ossie.yaml").read_text()
