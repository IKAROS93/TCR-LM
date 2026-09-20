uint8 CLOUD_CORNER=0
uint8 CLOUD_SURF=1
uint8 CLOUD_FULL=2

uint8 VALID_STABLE=0
uint8 CHANGE_CANDIDATE=1
uint8 CHANGED_NEW=2
uint8 STALE_OLD=3
uint8 REJECTED_NON_TERRAIN=4

uint32 keyframe_id
uint32 point_index
uint8 cloud_type

float32 x
float32 y
float32 z

float64 time
uint8 state
uint16 confirmations
float32 weight
