import ast
import pandas as pd

from training.team.inference_feature_builder import (
    prepare_inference_row,
    calculate_composition_strength,
    find_exact_composition,
)

AGENT_ROLE_MAP = {
    'iso':'duelist','jett':'duelist','raze':'duelist','reyna':'duelist','yoru':'duelist','neon':'duelist','phoenix':'duelist','waylay':'duelist',
    'breach':'initiator','fade':'initiator','gekko':'initiator','kayo':'initiator','skye':'initiator','sova':'initiator','tejo':'initiator',
    'astra':'controller','brimstone':'controller','clove':'controller','harbor':'controller','miks':'controller','omen':'controller','viper':'controller',
    'chamber':'sentinel','cypher':'sentinel','deadlock':'sentinel','killjoy':'sentinel','sage':'sentinel','veto':'sentinel','vyse':'sentinel',
}

PATH = 'training/dataset/valorant_dataset_team_v2.csv'
df = pd.read_csv(PATH)
df['Agent'] = df['Agent'].apply(ast.literal_eval)

# Historical rows: calculated Composition Strength must reproduce dataset value.
checks = [
    ('100 Thieves','Abyss',2024,['cypher','gekko','jett','omen','sova']),
    ('100 Thieves','Ascent',2024,['jett','kayo','killjoy','omen','sova']),
    ('100 Thieves','Bind',2024,['brimstone','fade','gekko','raze','viper']),
]

for team, map_name, year, agents in checks:
    result = calculate_composition_strength(df, team, map_name, year, agents, AGENT_ROLE_MAP)
    hist = find_exact_composition(df, team, map_name, year, agents).iloc[0]
    diff = abs(result['composition_strength'] - float(hist['Composition Strength']))
    print(team, map_name, year)
    print(' calculated:', result['composition_strength'])
    print(' dataset   :', float(hist['Composition Strength']))
    print(' diff      :', diff)
    assert diff <= 0.01

# Unseen composition: must not require a historical composition row.
team, map_name, year = '100 Thieves', 'Abyss', 2024
agents = ['breach','fade','jett','omen','sova']
exact = find_exact_composition(df, team, map_name, year, agents)
print('\nUNSEEN COMPOSITION')
print('historical rows:', len(exact))
assert exact.empty

row = prepare_inference_row(df, team, map_name, year, agents, AGENT_ROLE_MAP)
print(row[['Team','Map','Year','Agent','Role Pattern','Duelist Count','Initiator Count','Controller Count','Sentinel Count','Team Overall WR','Team Map WR','Composition Strength']].to_string(index=False))
assert row.loc[0, 'Composition Strength'] >= 0
assert row.loc[0, 'Team Overall WR'] == round(float(df[df.Team == team]['Winrate'].mean()), 4)
assert row.loc[0, 'Team Map WR'] == round(float(df[(df.Team == team)&(df.Map == map_name)]['Winrate'].mean()), 4)
print('[PASS] Historical strength reproduction')
print('[PASS] Unseen composition strength preparation')
