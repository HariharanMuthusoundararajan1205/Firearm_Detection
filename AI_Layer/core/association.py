from collections import defaultdict
import math

class Grid:
    def __init__(self,grid_size = 200):
        self.grid_size = grid_size

    def build_grid(self,persons):
        grid = defaultdict(list)

        for p in persons:
            x1,y1,x2,y2 = p["bbox"]
            cx,cy = (x1 + x2) // 2, (y1 + y2) // 2

            key = (cx // self.grid_size , cy // self.grid_size)
            grid[key].append(p)

        return grid
    
    def get_neighbors(self,key):
        x,y = key
        return [(x + dx, y + dy) for dx in [-1,0,1] for dy in [-1,0,1]]
    
    def center(self,bbox):
        x1,y1,x2,y2 = bbox
        return ((x1+x2)//2,(y1+y2)//2)
    
    def distance(self, c1,c2):
        return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    
    def associate(self, firearms, persons):
        grid = self.build_grid(persons)
        results = []

        for gun in firearms:
            gun_center = self.center(gun["bbox"])
            key = (gun_center[0] // self.grid_size, gun_center[1] // self.grid_size)

            candidates = []
            for k in self.get_neighbors(key):
                candidates.extend(grid.get(k, []))

            matched_id = None
            mathced_person_bbox = None
            min_dist = float("inf")

            for person in candidates:
                person_center = self.center(person["bbox"])
                dist = self.distance(gun_center, person_center)

                if dist < min_dist:
                    min_dist = dist
                    matched_id = person["track_id"]
                    mathced_person_bbox = person["bbox"]

            results.append({
                "bbox": gun["bbox"],
                "confidence": gun["confidence"],
                "Person_id": matched_id,
                "person_bbox" : mathced_person_bbox
            })
            print(f"[INFO] Person {matched_id} holds firearm")

        return results


