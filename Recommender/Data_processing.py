import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.decomposition import PCA

class DataProcessing():
    def prepare_data(self, travelers, trips, destinations):
        print("Prepare Data before Using for Model")
        travelers['TRAVEL_STYL'] = reduce_feature(travelers[['TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']])
        travelers['TRAVEL_MOTIVE'] = reduce_feature(travelers[['TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3']])

        # 노드 특성 준비
        print("Prepare Feature")
        traveler_features = torch.tensor([
            [t['GENDER'], t['AGE_GRP'], t['TRAVEL_STATUS_DESTINATION'], t['TRAVEL_STYL'], t['TRAVEL_MOTIVE']]
            for idx, t in travelers.iterrows()
        ], dtype=torch.float)

        trip_features = torch.tensor([
            [t['TRAVEL_PERIOD'],
            t['SHOPPING'],
            t['PARK'],
            t['HISTORY'],
            t['TOUR'],
            t['SPORTS'],
            t['ARTS'],
            t['PLAY'],
            t['CAMPING'],
            t['FESTIVAL'],
            t['SPA'],
            t['EDUCATION'],
            t['DRAMA'],
            t['PILGRIMAGE'],
            t['WELL'],
            t['SNS'],
            t['HOTEL'],
            t['NEWPLACE'],
            t['WITHPET'],
            t['MIMIC'],
            t['ECO'],
            t['HIKING']]
            for idx, t in trips.iterrows()
        ], dtype=torch.float)

        destination_features = torch.tensor([
            [d['Y'], d['M'], d['D'], d['S'], d['G'], d['X_COORD'], d['Y_COORD'],
            d['RESIDENCE_TIME_MIN'], d['VISIT_AREA_TYPE_CD'], d['REVISIT_YN'], d['VISIT_CHC_REASON_CD']]
            for idx, d in destinations.iterrows()
        ], dtype=torch.float)

        max_features = max(
            len(traveler_features[0]),
            len(trip_features[0]),
            len(destination_features[0])
        )

        # 노드 특성 준비 함수
        def prepare_features(data, max_features):
            features = []
            for item in data:
                feature = list(item)
                # 리스트 형태의 특성을 풀어서 추가
                feature = [val for sublist in feature for val in (sublist if isinstance(sublist, list) else [sublist])]
                # 부족한 특성을 0으로 채움
                feature += [0] * (max_features - len(feature))
                features.append(feature)
            return torch.tensor(features, dtype=torch.float)
        
        print("Prepare Node Feature")

        traveler_features = prepare_features(traveler_features, max_features)
        trip_features = prepare_features(trip_features, max_features)
        destination_features = prepare_features(destination_features, max_features)

        x = torch.cat([traveler_features, trip_features, destination_features], dim=0)
        
        print("Prepare Edge Feature")
        edge_index = create_edge_index(travelers, trips, destinations)

        # 목표값 준비
        print("Prepare Target")
        total_nodes = len(travelers) + len(trips) + len(destinations)
        y = torch.zeros((total_nodes, 3), dtype=torch.float)

        destination_start_idx = len(travelers) + len(trips)

        for idx, d in destinations.iterrows():
            y[destination_start_idx + idx] = torch.tensor([
                d.get('DGSTFN', 0), d.get('RCMDTN_INTENTION', 0), d.get('REVISIT_INTENTION', 0)
            ])

        mask = torch.zeros(total_nodes, dtype=torch.bool)
        mask[destination_start_idx:] = True

        return Data(x=x, edge_index=edge_index, y=y, mask=mask)

def create_edge_index(travelers, trips, destinations):
    edges = []

    # 여행자와 여행 간의 연결
    for _, trip in trips.iterrows():
        try:
            traveler_idx = travelers.index[travelers['TRAVELER_ID'] == trip['TRAVELER_ID']].tolist()[0]
            trip_idx = trips.index.tolist()[0] + len(travelers)
            edges.append([traveler_idx, trip_idx])
            edges.append([trip_idx, traveler_idx])  # 양방향 엣지
        except:
            continue

    # 여행과 여행지 간의 연결
    for _, destination in destinations.iterrows():
        try:
            trip_idx = trips.index[trips['TRAVEL_ID'] == destination['TRAVEL_ID']].tolist()[0] + len(travelers)
            dest_idx = destination.index.tolist()[0] + len(travelers) + len(trips)
            edges.append([trip_idx, dest_idx])
            edges.append([dest_idx, trip_idx])  # 양방향 엣지
        except:
            continue

    # 여행과 여행지 간의 연결
    for _, destination in destinations.iterrows():
        try:
            traveler_idx = travelers.index[travelers['TRAVELER_ID'] == trips.loc[trips['TRAVEL_ID'] == destination['TRAVEL_ID']]['TRAVELER_ID']].tolist()[0]
            dest_idx = destination.name + len(travelers) + len(trips)
            edges.append([traveler_idx, dest_idx])
            edges.append([dest_idx, traveler_idx]) # 양방향 엣지
        except:
            continue

    return torch.tensor(edges).t().contiguous()

def reduce_feature(data):
    pca = PCA(n_components=1)
    reduced_data = pca.fit_transform(data)

    return reduced_data