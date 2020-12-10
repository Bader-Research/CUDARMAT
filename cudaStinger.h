#if !defined(STINGER_H_)
#define STINGER_H_

#include <inttypes.h>

#define STINGER_MAX_LVERTICES (1L<<20)
#define STINGER_EDGEBLOCKSIZE 20    
#define STINGER_EDGEBLOCKS 2000000
#define STINGER_NUMETYPES 1

struct stinger_edge
{
     uint32_t neighbor;
     uint32_t weight;
     //int32_t timeFirst;
     //int32_t timeRecent;
};

struct stinger_eb
{
     struct stinger_eb * next;
     //int32_t etype;
     uint32_t vertexID;
     uint32_t numEdges;
     uint32_t high;
     //int32_t smallStamp;
     //int32_t largeStamp;
     struct stinger_edge edges[STINGER_EDGEBLOCKSIZE];
};

struct stinger_vb
{
     //int32_t physID;
     //int32_t vtype;
     //int32_t weight;
     //int32_t inDegree;
     uint32_t outDegree;
     struct stinger_eb * edges;
};

struct stinger_etype_array
{
     uint32_t length;
     uint32_t high;
     //struct stinger_eb ** blocks;
	  struct stinger_eb blocks[STINGER_EDGEBLOCKS];
};

//struct stinger_physmap;

struct stinger
{
     //struct stinger_physmap *physmap;
     struct stinger_vb * LVA;
     struct stinger_etype_array ETA[STINGER_NUMETYPES];
     uint32_t N, maxvtx;
     //int32_t LVASize;
};

void stinger_init (void);
struct stinger * stinger_new (void);
struct stinger * stinger_free (struct stinger*);
struct stinger * stinger_free_all (struct stinger*);

#if 0
int32_t stinger_activate_vtx (struct stinger*, int32_t /* physical id */,
			      int32_t /* VType */,
			      int32_t /* weight */);
void stinger_activate_vtx_range (struct stinger*,
				 int32_t, int32_t /* [start, end) */,
				 int32_t /* VType */);
int stinger_set_vtx (struct stinger*, int32_t /* physical id */,
		     int32_t /* VType */, int32_t /* weight */);

int stinger_insert_edge (struct stinger*,
			 int32_t /* type */,
			 int32_t /* from */, int32_t /* to */,
			 int32_t /* int weight */,
			 int32_t /* timestamp */);
int stinger_insert_edge_pair (struct stinger*,
			      int32_t /* type */,
			      int32_t /* from */, int32_t /* to */,
			      int32_t /* int weight */,
			      int32_t /* timestamp */);
int stinger_incr_edge (struct stinger*,
		       int32_t /* type */,
		       int32_t /* from */, int32_t /* to */,
		       int32_t /* int weight */,
		       int32_t /* timestamp */);
int stinger_incr_edge_pair (struct stinger*,
			    int32_t /* type */,
			    int32_t /* from */, int32_t /* to */,
			    int32_t /* int weight */,
			    int32_t /* timestamp */);
int stinger_remove_edge (struct stinger*,
			 int32_t /* type */,
			 int32_t /* from */, int32_t /* to */);
int stinger_remove_edge_pair (struct stinger*,
			      int32_t /* type */,
			      int32_t /* from */, int32_t /* to */);
int stinger_remove_and_insert_edges (struct stinger *,
				     int32_t /* EType */,
				     int32_t /* from */,
				     int32_t /* nremove */, int32_t * /* to remove */,
				     int32_t /* ninsert */, int32_t * /* to insert */,
				     int32_t * /* weight or NULL for 1 */,
				     int32_t /* timestamp */);

void stinger_set_initial_edges (struct stinger * /* G */,
				const size_t /* nv */,
				const int32_t /* EType */,
				const int32_t * /* off */,
				const int32_t * /* phys_to */,
				const int32_t * /* weights */,
				const int32_t * /* times */,
				const int32_t * /* first_times */,
				const int32_t /* single_ts, if !times or !first_times */);

void stinger_gather_typed_successors (const struct stinger*,
				      int32_t /* type */, int32_t /* vtx */,
				      size_t* /* outlen */, int32_t* /* out */,
				      size_t /* max_outlen */);
int stinger_has_typed_successor (const struct stinger*,
				 int32_t, int32_t, int32_t);
#endif

static inline int32_t stinger_nvtx_max (const struct stinger*);
static inline int32_t stinger_nvtx_max_active (const struct stinger*);

int32_t
stinger_nvtx_max (const struct stinger *S_ /*UNUSED*/)
{
     return STINGER_MAX_LVERTICES;
}

int32_t
stinger_nvtx_max_active (const struct stinger *S_)
{
     return 1+S_->maxvtx;
}

static inline int32_t stinger_outdegree (const struct stinger*, int32_t);
//static inline int32_t stinger_indegree (const struct stinger*, int32_t);
//static inline int32_t stinger_physind (const struct stinger*, int32_t);
//static inline int32_t stinger_vweight (const struct stinger*, int32_t);
//static inline int32_t stinger_vtype (const struct stinger*, int32_t);
static inline const struct stinger_eb* stinger_edgeblocks (const struct stinger*, int32_t);

static inline const struct stinger_eb* stinger_next_eb (const struct stinger*, const struct stinger_eb *);
static inline int stinger_eb_high (const struct stinger_eb*);
//static inline int32_t stinger_eb_type (const struct stinger_eb*);

static inline int stinger_eb_is_blank (const struct stinger_eb*, int);
static inline int32_t stinger_eb_adjvtx (const struct stinger_eb*, int);
static inline int32_t stinger_eb_weight (const struct stinger_eb*, int);
//static inline int32_t stinger_eb_ts (const struct stinger_eb*, int);
//static inline int32_t stinger_eb_first_ts (const struct stinger_eb*, int);

int32_t
stinger_outdegree (const struct stinger *S_, int32_t i_)
{
     return S_->LVA[i_].outDegree;
}

#if 0
int32_t
stinger_indegree (const struct stinger *S_, int32_t i_)
{
     return S_->LVA[i_].inDegree;
}

int32_t
stinger_physind (const struct stinger *S_, int32_t i_)
{
     return S_->LVA[i_].physID;
}

int32_t
stinger_vweight (const struct stinger *S_, int32_t i_)
{
     return S_->LVA[i_].weight;
}

int32_t
stinger_vtype (const struct stinger *S_, int32_t i_)
{
     return S_->LVA[i_].vtype;
}
#endif

const struct stinger_eb*
stinger_edgeblocks (const struct stinger *S_, int32_t i_)
{
     return S_->LVA[i_].edges;
}

const struct stinger_eb*
stinger_next_eb (const struct stinger *G /*UNUSED*/, const struct stinger_eb *eb_)
{
     return eb_->next;
}
      
#if 0	
int32_t
stinger_eb_type (const struct stinger_eb *eb_)
{
     return eb_->etype;
}
#endif

int
stinger_eb_high (const struct stinger_eb *eb_)
{
     return eb_->high;
}

int
stinger_eb_is_blank (const struct stinger_eb *eb_, int k_)
{
     return eb_->edges[k_].weight == 0;
}

int32_t
stinger_eb_adjvtx (const struct stinger_eb *eb_, int k_)
{
     return eb_->edges[k_].neighbor;
}

int32_t
stinger_eb_weight (const struct stinger_eb *eb_, int k_)
{
     return eb_->edges[k_].weight;
}

#if 0
int32_t
stinger_eb_ts (const struct stinger_eb *eb_, int k_)
{
     return eb_->edges[k_].timeRecent;
}

int32_t
stinger_eb_first_ts (const struct stinger_eb *eb_, int k_)
{
     return eb_->edges[k_].timeFirst;
}

int32_t stinger_count_outdeg (struct stinger * G, int32_t v);

struct stinger_patch {
  int nvtx;
  int32_t * restrict vtx;
  int * restrict off;
  int32_t * restrict adj;
  int * restrict incr;
};

void stinger_patch_free (struct stinger_patch *);
/* int chosen deliberately. Patches shouldn't be too large. */
void stinger_patch_alloc (struct stinger_patch *, const int /*nvtx*/, const int /*nedge*/);
struct stinger_patch stinger_batch_to_patch_ws (const int, int32_t *, struct stinger_patch, int*);
struct stinger_patch stinger_batch_to_patch (const int, int32_t *);
/* does *not* form the transpose: */
struct stinger_patch stinger_batch_to_patch_undirected_ws (const int, int32_t *, struct stinger_patch, int*);
struct stinger_patch stinger_batch_to_patch_undirected (const int, int32_t *);

#define STINGER_FORALL_EB_BEGIN(STINGER_,STINGER_SRCVTX_,STINGER_EBNM_)	\
do {									\
  const struct stinger * stinger__ = (STINGER_);			\
  const int32_t stinger_srcvtx__ = (STINGER_SRCVTX_);			\
  if (stinger_physind (stinger__, stinger_srcvtx__) >= 0) {		\
    const struct stinger_eb * restrict stinger_eb__;			\
    stinger_eb__ = stinger_edgeblocks (stinger__, stinger_srcvtx__);	\
    while (stinger_eb__) {						\
      const struct stinger_eb * restrict STINGER_EBNM_ = stinger_eb__;	\
      do
#define STINGER_FORALL_EB_END()					\
      while (0);						\
      stinger_eb__ = stinger_next_eb (stinger__, stinger_eb__);	\
    }								\
  }								\
} while (0)
#endif


#endif /* STINGER_H_ */
